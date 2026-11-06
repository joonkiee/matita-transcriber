// Matita.app: a small native shell around the Matita server (matita.py).
// It starts the bundled Python server, shows the page in its own window, and stops the server on quit.
// Build: swiftc -O -target arm64-apple-macos14.0 app/Matita.swift -o Matita   (see scripts/build-app.sh)

import Cocoa
import WebKit
import ServiceManagement

let port = 8765
let home = URL(string: "http://127.0.0.1:\(port)/")!

// MARK: leftovers from an earlier run

/// Runs a small command and returns what it printed.
func runQuiet(_ path: String, _ args: [String]) -> String {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: path)
    p.arguments = args
    let out = Pipe()
    p.standardOutput = out
    p.standardError = FileHandle.nullDevice
    do { try p.run() } catch { return "" }
    let data = out.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    return String(data: data, encoding: .utf8) ?? ""
}

/// Programs listening on Matita's port that are a Matita server (this app's, an older app's,
/// or the old Terminal install), and not this process.
func matitaServersOnPort() -> [pid_t] {
    let pids = runQuiet("/usr/sbin/lsof", ["-nP", "-t", "-iTCP:\(port)", "-sTCP:LISTEN"])
        .split(whereSeparator: \.isNewline).compactMap { pid_t($0.trimmingCharacters(in: .whitespaces)) }
    return pids.filter { pid in
        guard pid != getpid() else { return false }
        let cmd = runQuiet("/bin/ps", ["-o", "command=", "-p", String(pid)])
        return cmd.contains("matita.py") || cmd.contains("stt_app.py")
    }
}

/// A Matita server left running after the app quit unexpectedly (or from an older copy) holds the
/// port, and a new one can't start. Ask it to stop, then make it stop.
func clearStaleServers() {
    var left = matitaServersOnPort()
    guard !left.isEmpty else { return }
    for pid in left {
        kill(pid, SIGCONT)          // in case it was paused
        kill(pid, SIGTERM)          // it saves its queue and exits
    }
    for _ in 0..<30 {
        usleep(100_000)
        left = matitaServersOnPort()
        if left.isEmpty { return }
    }
    for pid in left { kill(pid, SIGKILL) }
    usleep(300_000)
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKUIDelegate, WKNavigationDelegate, WKDownloadDelegate, NSWindowDelegate {
    var window: NSWindow!
    var web: WKWebView!
    var loading: NSTextField!
    var server: Process?
    var quitting = false
    var restarts = 0

    // MARK: launch

    func applicationDidFinishLaunching(_ note: Notification) {
        buildMenu()
        makeWindow()
        DispatchQueue.global(qos: .userInitiated).async {
            clearStaleServers()
            DispatchQueue.main.async {
                self.startServer()
                self.waitForServer(tries: 0)
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { false }   // keeps transcribing

    func applicationShouldHandleReopen(_ app: NSApplication, hasVisibleWindows: Bool) -> Bool {
        window.makeKeyAndOrderFront(nil)
        return true
    }

    func applicationWillTerminate(_ note: Notification) {
        quitting = true
        guard let p = server, p.isRunning else { return }
        p.terminate()                                   // SIGTERM: Matita saves the queue and finishes pending trash
        let deadline = Date().addingTimeInterval(6)
        while p.isRunning && Date() < deadline { usleep(100_000) }
        if p.isRunning { kill(p.processIdentifier, SIGKILL) }
    }

    // MARK: server

    func startServer() {
        guard let res = Bundle.main.resourceURL else { return }
        let p = Process()
        p.executableURL = res.appendingPathComponent("python/bin/python3")
        p.arguments = [res.appendingPathComponent("matita.py").path, "--serve"]
        var env = ProcessInfo.processInfo.environment
        env["MATITA_APP_BUNDLE"] = Bundle.main.bundlePath
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        p.environment = env
        let logDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("whisper-app")
        try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        let logURL = logDir.appendingPathComponent("matita.log")
        if !FileManager.default.fileExists(atPath: logURL.path) { FileManager.default.createFile(atPath: logURL.path, contents: nil) }
        if let log = try? FileHandle(forWritingTo: logURL) {
            log.seekToEndOfFile()
            p.standardOutput = log
            p.standardError = log
        }
        p.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                guard let self = self, !self.quitting else { return }
                // restart a crashed server a few times; port-in-use (exit 1) means another copy is running
                if self.restarts < 3 && proc.terminationStatus != 1 {
                    self.restarts += 1
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) { self.startServer() }
                }
            }
        }
        do { try p.run(); server = p } catch { showLoading("Matita couldn't start: \(error.localizedDescription)") }
    }

    func waitForServer(tries: Int) {
        var req = URLRequest(url: home.appendingPathComponent("api/info"))
        req.timeoutInterval = 1
        URLSession.shared.dataTask(with: req) { data, _, _ in
            var ours = false
            if let d = data, let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
                ours = (j["app"] as? Bool) == true
            }
            DispatchQueue.main.async {
                if ours {
                    self.loading.isHidden = true
                    self.web.isHidden = false
                    self.web.load(URLRequest(url: home))
                } else if tries < 240 {
                    if tries == 20 { self.showLoading("Starting Matita…") }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { self.waitForServer(tries: tries + 1) }
                } else {
                    self.showLoading("Matita didn't start. See ~/whisper-app/matita.log, or quit and open it again.")
                }
            }
        }.resume()
    }

    // MARK: window

    func makeWindow() {
        let cfg = WKWebViewConfiguration()
        cfg.mediaTypesRequiringUserActionForPlayback = []
        cfg.preferences.javaScriptCanOpenWindowsAutomatically = false
        cfg.preferences.setValue(true, forKey: "developerExtrasEnabled")     // right-click → Inspect, for bug reports
        web = WKWebView(frame: .zero, configuration: cfg)
        web.uiDelegate = self
        web.navigationDelegate = self
        web.allowsBackForwardNavigationGestures = false
        web.setValue(false, forKey: "drawsBackground")                       // the window colour shows while loading
        web.isHidden = true

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1120, height: 820),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Matita"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.minSize = NSSize(width: 720, height: 560)
        window.backgroundColor = NSColor(name: nil) { a in
            a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(srgbRed: 0x15/255.0, green: 0x15/255.0, blue: 0x11/255.0, alpha: 1)
                : NSColor(srgbRed: 0xf4/255.0, green: 0xf2/255.0, blue: 0xea/255.0, alpha: 1)
        }
        window.delegate = self
        window.isReleasedWhenClosed = false
        window.setFrameAutosaveName("MatitaMain")
        if !window.setFrameUsingName("MatitaMain") { window.center() }

        let box = NSView()
        loading = NSTextField(labelWithString: "")
        loading.textColor = .secondaryLabelColor
        loading.alignment = .center
        loading.isHidden = true
        loading.translatesAutoresizingMaskIntoConstraints = false
        web.translatesAutoresizingMaskIntoConstraints = false
        box.addSubview(web)
        box.addSubview(loading)
        NSLayoutConstraint.activate([
            web.leadingAnchor.constraint(equalTo: box.leadingAnchor), web.trailingAnchor.constraint(equalTo: box.trailingAnchor),
            web.topAnchor.constraint(equalTo: box.topAnchor), web.bottomAnchor.constraint(equalTo: box.bottomAnchor),
            loading.centerXAnchor.constraint(equalTo: box.centerXAnchor), loading.centerYAnchor.constraint(equalTo: box.centerYAnchor),
            loading.widthAnchor.constraint(lessThanOrEqualTo: box.widthAnchor, constant: -60),
        ])
        window.contentView = box
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func showLoading(_ text: String) {
        loading.stringValue = text
        loading.isHidden = false
    }

    // MARK: menus

    func buildMenu() {
        let main = NSMenu()
        func item(_ title: String, _ sel: Selector?, _ key: String = "", _ mods: NSEvent.ModifierFlags = .command) -> NSMenuItem {
            let i = NSMenuItem(title: title, action: sel, keyEquivalent: key)
            i.keyEquivalentModifierMask = mods
            return i
        }
        let appMenu = NSMenu()
        appMenu.addItem(item("About Matita", #selector(NSApplication.orderFrontStandardAboutPanel(_:))))
        appMenu.addItem(.separator())
        appMenu.addItem(item("Settings…", #selector(openSettings), ","))
        appMenu.addItem(item("Check for Updates…", #selector(checkUpdates)))
        let login = item("Open at Login", #selector(toggleLogin))
        login.state = SMAppService.mainApp.status == .enabled ? .on : .off
        appMenu.addItem(login)
        appMenu.addItem(.separator())
        appMenu.addItem(item("Hide Matita", #selector(NSApplication.hide(_:)), "h"))
        appMenu.addItem(item("Hide Others", #selector(NSApplication.hideOtherApplications(_:)), "h", [.command, .option]))
        appMenu.addItem(item("Show All", #selector(NSApplication.unhideAllApplications(_:))))
        appMenu.addItem(.separator())
        appMenu.addItem(item("Quit Matita", #selector(NSApplication.terminate(_:)), "q"))
        let a = NSMenuItem(); a.submenu = appMenu; main.addItem(a)

        let file = NSMenu(title: "File")
        file.addItem(item("Add Files…", #selector(addFiles), "o"))
        file.addItem(item("Close Window", #selector(NSWindow.performClose(_:)), "w"))
        let f = NSMenuItem(); f.submenu = file; main.addItem(f)

        let edit = NSMenu(title: "Edit")                // without these, ⌘C / ⌘V don't reach the page
        edit.addItem(item("Undo", Selector(("undo:")), "z"))
        edit.addItem(item("Redo", Selector(("redo:")), "z", [.command, .shift]))
        edit.addItem(.separator())
        edit.addItem(item("Cut", #selector(NSText.cut(_:)), "x"))
        edit.addItem(item("Copy", #selector(NSText.copy(_:)), "c"))
        edit.addItem(item("Paste", #selector(NSText.paste(_:)), "v"))
        edit.addItem(item("Select All", #selector(NSText.selectAll(_:)), "a"))
        let e = NSMenuItem(); e.submenu = edit; main.addItem(e)

        let view = NSMenu(title: "View")
        view.addItem(item("Reload", #selector(reload), "r"))
        view.addItem(item("Enter Full Screen", #selector(NSWindow.toggleFullScreen(_:)), "f", [.command, .control]))
        let v = NSMenuItem(); v.submenu = view; main.addItem(v)

        let win = NSMenu(title: "Window")
        win.addItem(item("Minimize", #selector(NSWindow.performMiniaturize(_:)), "m"))
        win.addItem(item("Zoom", #selector(NSWindow.performZoom(_:))))
        win.addItem(item("Matita", #selector(showMain), "0"))
        let w = NSMenuItem(); w.submenu = win; main.addItem(w)
        NSApp.windowsMenu = win

        let help = NSMenu(title: "Help")
        help.addItem(item("Matita on GitHub", #selector(openGitHub)))
        help.addItem(item("Show Log File", #selector(showLog)))
        let h = NSMenuItem(); h.submenu = help; main.addItem(h)
        NSApp.helpMenu = help
        NSApp.mainMenu = main
    }

    func js(_ code: String) { showMain(); web.evaluateJavaScript(code, completionHandler: nil) }
    @objc func openSettings() { js("document.getElementById('setOpen') && document.getElementById('setOpen').click()") }
    @objc func checkUpdates() { js("document.getElementById('chk') && document.getElementById('chk').click()") }
    @objc func addFiles() { js("document.getElementById('choose') && document.getElementById('choose').click()") }
    @objc func reload() { web.reload() }
    @objc func showMain() { window.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true) }
    @objc func openGitHub() { NSWorkspace.shared.open(URL(string: "https://github.com/joonkiee/matita-transcriber")!) }
    @objc func showLog() {
        let log = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("whisper-app/matita.log")
        NSWorkspace.shared.activateFileViewerSelecting([log])
    }
    @objc func toggleLogin(_ sender: NSMenuItem) {
        do {
            if SMAppService.mainApp.status == .enabled { try SMAppService.mainApp.unregister() }
            else { try SMAppService.mainApp.register() }
        } catch {
            let a = NSAlert(); a.messageText = "Couldn't change Open at Login"; a.informativeText = error.localizedDescription; a.runModal()
        }
        sender.state = SMAppService.mainApp.status == .enabled ? .on : .off
    }

    // MARK: web view

    // the microphone for Record (macOS asks once for Matita)
    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin, initiatedByFrame frame: WKFrameInfo,
                 type: WKMediaCaptureType, decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        decisionHandler(origin.host == "127.0.0.1" || origin.host == "localhost" ? .grant : .deny)
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.canChooseFiles = true
        panel.beginSheetModal(for: window) { r in completionHandler(r == .OK ? panel.urls : nil) }
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping () -> Void) {
        let a = NSAlert(); a.messageText = message; a.addButton(withTitle: "OK")
        a.beginSheetModal(for: window) { _ in completionHandler() }
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping (Bool) -> Void) {
        let a = NSAlert(); a.messageText = message; a.addButton(withTitle: "OK"); a.addButton(withTitle: "Cancel")
        a.beginSheetModal(for: window) { r in completionHandler(r == .alertFirstButtonReturn) }
    }

    // links that open a new window (GitHub, Ollama, Anthropic…) go to the default browser
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for navigationAction: WKNavigationAction,
                 windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let u = navigationAction.request.url { NSWorkspace.shared.open(u) }
        return nil
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if action.shouldPerformDownload { return decisionHandler(.download) }
        if let u = action.request.url, let host = u.host, host != "127.0.0.1" && host != "localhost",
           u.scheme == "http" || u.scheme == "https" {
            NSWorkspace.shared.open(u)
            return decisionHandler(.cancel)
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse, decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        let attach = ((response.response as? HTTPURLResponse)?.value(forHTTPHeaderField: "Content-Disposition") ?? "").lowercased().hasPrefix("attachment")
        decisionHandler(attach || !response.canShowMIMEType ? .download : .allow)
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) { download.delegate = self }
    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) { download.delegate = self }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) { webView.reload() }

    // downloads (Save as…, Download all) land in ~/Downloads and are shown in Finder
    var pending: [ObjectIdentifier: URL] = [:]
    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String,
                  completionHandler: @escaping (URL?) -> Void) {
        let dir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        let base = (suggestedFilename as NSString).deletingPathExtension, ext = (suggestedFilename as NSString).pathExtension
        var url = dir.appendingPathComponent(suggestedFilename), n = 2
        while FileManager.default.fileExists(atPath: url.path) {
            url = dir.appendingPathComponent(ext.isEmpty ? "\(base) (\(n))" : "\(base) (\(n)).\(ext)"); n += 1
        }
        pending[ObjectIdentifier(download)] = url
        completionHandler(url)
    }
    func downloadDidFinish(_ download: WKDownload) {
        if let u = pending.removeValue(forKey: ObjectIdentifier(download)) { NSWorkspace.shared.activateFileViewerSelecting([u]) }
    }
    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        pending.removeValue(forKey: ObjectIdentifier(download))
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
