#!/usr/bin/env python3
"""Writes Matita's guide and comparison pages into docs/, plus docs/sitemap.xml.

    python3 scripts/build-pages.py

Each page is plain HTML sharing docs/page.css. Facts about other products are dated and linked;
re-check them before changing the date in CHECKED.
"""
import html, json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
BASE = "https://joonkiee.github.io/matita-transcriber/"
DL = "https://github.com/joonkiee/matita-transcriber/releases/latest"
GH = "https://github.com/joonkiee/matita-transcriber"
CHECKED = "5 November 2026"
CHECKED_KO = "2026년 11월 5일"

LOGO = ('<svg viewBox="0 0 100 100" aria-hidden="true"><rect width="100" height="100" rx="23" fill="#1f4d35"/>'
        '<path d="M27 70V31L50 56L73 31V70" fill="none" stroke="#a6e8ae" stroke-width="9" stroke-linecap="round" '
        'stroke-linejoin="round"/></svg>')

SRC = {
    "otter": '<a href="https://otter.ai/pricing">Otter pricing</a>',
    "granola": '<a href="https://www.granola.ai/pricing">Granola pricing</a>, '
               '<a href="https://docs.granola.ai/help-center/taking-notes/transcription">Granola help</a>',
}

EN_NAV = {"home": "Home", "guides": "Guides", "otter": "Otter alternative", "granola": "Granola alternative",
          "whisper": "Whisper for Mac", "lang": "한국어", "dl": "Download"}
KO_NAV = {"home": "홈", "guides": "가이드", "otter": "Otter 대안", "granola": "Granola 대안",
          "whisper": "Mac용 Whisper", "lang": "English", "dl": "다운로드"}

PAGES = []          # (path, lang) for the sitemap


def page(path, *, lang, title, desc, h1, lead, body, crumb, related, faq=None, alt=None, cta=None):
    """path is relative to docs/, ending in '/'."""
    depth = path.count("/")
    up = "../" * depth
    nav = KO_NAV if lang == "ko" else EN_NAV
    url = BASE + path
    home = up + ("ko/" if lang == "ko" else "")
    other = alt if alt is not None else ("" if lang == "ko" else "ko/")
    ld = [{"@context": "https://schema.org", "@type": "Article", "headline": h1, "description": desc,
           "inLanguage": lang, "url": url, "image": BASE + "og.png",
           "author": {"@type": "Person", "name": "Jun Ki Shin", "url": "https://github.com/joonkiee"},
           "about": {"@type": "SoftwareApplication", "name": "Matita", "operatingSystem": "macOS 14 or later",
                     "applicationCategory": "BusinessApplication", "url": BASE,
                     "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}}},
          {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
              {"@type": "ListItem", "position": i + 1, "name": n, "item": BASE + p} for i, (n, p) in enumerate(crumb)]}]
    if faq:
        ld.append({"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in faq]})
        body += ("<h2>%s</h2>" % ("자주 묻는 질문" if lang == "ko" else "Questions")) + "".join(
            "<h3>%s</h3><p>%s</p>" % (html.escape(q), html.escape(a)) for q, a in faq)
    crumbs = " › ".join('<a href="%s%s">%s</a>' % (up, p, html.escape(n)) for n, p in crumb[:-1])
    rel = "".join('<li><a href="%s%s">%s</a></li>' % (up, p, html.escape(n)) for n, p in related)
    cta = cta or (("무료로 받아 보세요", "Apple Silicon Mac, macOS 14 이상. 계정 없이 바로 써요.", "Mac용 다운로드")
                  if lang == "ko" else
                  ("Try it on your next recording", "Free for Apple Silicon Macs on macOS 14 or later. No account.",
                   "Download for Mac"))
    hreflang = ""
    if alt is not None or path in ("", "ko/"):
        pass
    if alt:
        en, ko = (alt, path) if lang == "ko" else (path, alt)
        hreflang = ('<link rel="alternate" hreflang="en" href="%s%s">\n<link rel="alternate" hreflang="ko" href="%s%s">\n'
                    % (BASE, en, BASE, ko))
    out = f"""<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(desc)}">
<link rel="canonical" href="{url}">
{hreflang}<meta property="og:type" content="article">
<meta property="og:site_name" content="Matita">
<meta property="og:url" content="{url}">
<meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(desc)}">
<meta property="og:image" content="{BASE}og.png">
<meta property="og:image:width" content="1200"><meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0e0f0e" media="(prefers-color-scheme: dark)">
<link rel="icon" href="{up}icon.png">
<link rel="stylesheet" href="{up}page.css">
<script type="application/ld+json">{json.dumps(ld, ensure_ascii=False)}</script>
</head>
<body>
<header class="top"><div class="w">
  <nav>
    <a class="brand" href="{home}">{LOGO}Matita</a>
    <div class="links"><a href="{up}{'ko/' if lang == 'ko' else ''}guides/">{nav['guides']}</a><a href="{up}otter-alternative/">{nav['otter']}</a><a href="{up}whisper-mac-app/">{nav['whisper']}</a><a href="{up}{other}" lang="{'en' if lang == 'ko' else 'ko'}">{nav['lang']}</a></div>
    <a class="btn dark" href="{DL}">{nav['dl']}</a>
  </nav>
  <div class="crumb">{crumbs}</div>
  <h1>{h1}</h1>
  <p class="lead">{lead}</p>
</div></header>
<main>
<article>
{body}
</article>
<div class="end"><div class="in"><h2>{cta[0]}</h2><p>{cta[1]}</p><a class="btn dark" href="{DL}">{cta[2]}</a></div></div>
<div class="related"><h2>{'더 읽어 보기' if lang == 'ko' else 'Related'}</h2><ul>{rel}</ul></div>
</main>
<footer><div class="w"><span>Matita · MIT licence · {'Otter, Granola, Wispr Flow와 관련 없는 개인 프로젝트예요.' if lang == 'ko' else 'Not affiliated with Otter, Granola or Wispr.'}</span><span><a href="{GH}">GitHub</a> · <a href="{GH}/issues/new/choose">{'문제 신고' if lang == 'ko' else 'Report a problem'}</a></span></div></footer>
</body>
</html>
"""
    d = os.path.join(ROOT, path)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
        f.write(out)
    PAGES.append((path, lang))


def fig(src, alt, cap, depth):
    up = "../" * depth
    return (f'<figure><picture><source srcset="{up}site/{src}-dark.webp" media="(prefers-color-scheme: dark)">'
            f'<img src="{up}site/{src}-light.webp" alt="{html.escape(alt)}" loading="lazy" width="2360" height="1520">'
            f'</picture><figcaption>{cap}</figcaption></figure>')


def table(rows, heads):
    th = "".join('<th class="mc">%s</th>' % heads[0] if i == 0 else "<th>%s</th>" % h for i, h in enumerate(heads))
    tr = "".join("<tr><th>%s</th><td class=\"mc\">%s</td>%s</tr>" % (r[0], r[1], "".join("<td>%s</td>" % c for c in r[2:]))
                 for r in rows)
    return f'<div class="tbl"><table><thead><tr><th></th>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


# ------------------------------------------------------------------ comparison pages
otter_rows = [
    ("Where speech becomes text", "On your Mac", "Otter's cloud"),
    ("Files you already have", "Any audio or video file", "3 imports on the free plan"),
    ("Free plan", "Everything, no limits", "300 minutes a month, 30 minutes per conversation"),
    ("Paid plan", "None", "$16.99 a month, or $8.33 a month billed yearly"),
    ("Meetings", "Records both sides from your Mac. No bot joins.", "A bot can join the call"),
    ("Account", "Not needed", "Required"),
]
page("otter-alternative/", lang="en",
     title="Free Otter alternative for Mac that works offline · Matita",
     desc="Matita is a free, open-source Otter alternative for Mac. Transcribe meetings, lectures and audio files on your Mac, with no minute limits and no account.",
     h1="A free, offline Otter alternative for Mac",
     lead="Matita transcribes meetings, lectures and the recordings you already have, on your own Mac. No minute limits, no upload, no account.",
     crumb=[("Matita", ""), ("Otter alternative", "otter-alternative/")],
     related=[("Granola alternative", "granola-alternative/"), ("Record a Zoom call without a bot", "guides/record-zoom-without-a-bot/"),
              ("Whisper for Mac", "whisper-mac-app/")],
     body=f"""
<p>Otter is a cloud service: your audio goes to Otter's servers and your transcripts live in Otter's app. Matita does the same job on your Mac. It's free and open source, and every transcript is a plain file in a folder you choose.</p>
<h2>Matita and Otter side by side</h2>
{table(otter_rows, ["Matita", "Otter"])}
<p class="note">Otter's details as listed on {SRC['otter']} on {CHECKED}. Prices in US dollars, per user, before tax.</p>
<h2>When Matita is the better fit</h2>
<ul>
<li><b>You'd rather your audio didn't leave your Mac.</b> Speech recognition runs on your Mac with whisper.cpp or NVIDIA Parakeet. Summaries can run on your Mac too, with Ollama.</li>
<li><b>You transcribe a lot.</b> There are no monthly minutes and no length limit per recording.</li>
<li><b>You have recordings already.</b> Drop in any audio or video file, or let Matita watch a folder.</li>
<li><b>You keep notes in files.</b> Transcripts are .txt and .srt files; you can save Word or Markdown, or write notes straight into Obsidian.</li>
</ul>
<h2>When Otter is the better fit</h2>
<ul>
<li>You need it on Windows, Android or in a browser. Matita runs only on Macs with Apple Silicon.</li>
<li>You want transcripts shared with your team in one online workspace.</li>
<li>You'd like a bot that joins your calendar's meetings for you.</li>
</ul>
<h2>Getting started</h2>
<ol class="steps">
<li><b>Download Matita</b> and drag it into Applications.</li>
<li><b>Open it once.</b> Matita isn't signed with a paid Apple Developer ID, so macOS asks you to confirm: click Done, then System Settings → Privacy &amp; Security → Open Anyway.</li>
<li><b>Add a recording</b> (or press ⌘O), or click <b>Record</b>. The first time, Matita downloads a speech model (640 MB to 1.6 GB).</li>
</ol>
""",
     faq=[("Is Matita really free?", "Yes. Matita is open source under the MIT licence. There's no paid plan, no account and no usage limit."),
          ("Does Matita need an internet connection?", "Only to download a speech model the first time, to check for updates, and, if you choose Claude for summaries, to send the transcript to Anthropic. Transcription itself works offline."),
          ("Can Matita record Zoom, Teams or Google Meet?", "Yes, on macOS 14.2 or later. Turn on “Include sound from other apps” when recording, and Matita records your microphone and the other people in the call. No bot joins.")])

granola_rows = [
    ("Where speech becomes text", "On your Mac", "A cloud transcription provider"),
    ("The original audio", "Stays on your Mac", "Not kept"),
    ("Files you already have", "Any audio or video file", "Not supported"),
    ("Free plan", "Everything, no limits", "Meeting history for 30 days"),
    ("Paid plan", "None", "$14 per user a month"),
    ("Meetings", "No bot. Records both sides from your Mac.", "No bot"),
    ("Account", "Not needed", "Required"),
]
page("granola-alternative/", lang="en",
     title="Free Granola alternative that keeps meetings on your Mac · Matita",
     desc="Matita is a free, open-source Granola alternative for Mac. Bot-free meeting transcripts and AI summaries made on your Mac, plus your existing audio files.",
     h1="A free Granola alternative that keeps meetings on your Mac",
     lead="Like Granola, Matita records meetings without a bot. Unlike Granola, it transcribes on your Mac, keeps the audio, and works with recordings you already have.",
     crumb=[("Matita", ""), ("Granola alternative", "granola-alternative/")],
     related=[("Otter alternative", "otter-alternative/"), ("Record a Zoom call without a bot", "guides/record-zoom-without-a-bot/"),
              ("Transcribe iPhone Voice Memos on your Mac", "guides/transcribe-voice-memos-on-mac/")],
     body=f"""
<p>Granola and Matita both listen to your meetings from your computer instead of sending a bot. The difference is where the work happens. Granola sends the audio to a cloud transcription provider and doesn't keep it. Matita transcribes on your Mac and leaves the audio and transcript in your folders.</p>
<h2>Matita and Granola side by side</h2>
{table(granola_rows, ["Matita", "Granola"])}
<p class="note">Granola's details as listed on {SRC['granola']} on {CHECKED}. Prices in US dollars, before tax.</p>
<h2>When Matita is the better fit</h2>
<ul>
<li><b>You want to go back to the audio.</b> Every line has a timestamp; click it to hear that moment.</li>
<li><b>You record more than meetings.</b> Lectures, interviews, iPhone Voice Memos and old files all go through the same app.</li>
<li><b>You want everything, for good.</b> No history limit, and transcripts are plain files you can open anywhere.</li>
<li><b>You'd rather nothing went to the cloud.</b> Use Ollama for summaries and nothing leaves your Mac.</li>
</ul>
<h2>When Granola is the better fit</h2>
<ul>
<li>You like writing your own notes during the meeting and having AI fill them in. That's Granola's core idea; Matita summarises after the recording.</li>
<li>You need Windows or iPhone, or want to share notes with your team in one place.</li>
</ul>
{fig("speakers", "Matita's Speakers view, with each person's lines grouped", "Matita's Speakers view collects what each person said, with their share of the talking.", 1)}
""",
     faq=[("Does Matita join my meetings as a bot?", "No. It records your microphone and, on macOS 14.2 or later, the sound of the call app on your Mac."),
          ("Can Matita tell who said what?", "Yes. Turn on Speakers before transcribing. Choosing the actual number of people works better than Auto."),
          ("Are AI summaries private?", "With Ollama, summaries are made on your Mac. If you choose Claude instead, the transcript is sent to Anthropic to write the summary.")])

page("whisper-mac-app/", lang="en",
     title="Free Whisper app for Mac, no Terminal needed · Matita",
     desc="Run OpenAI's Whisper (large-v3-turbo) on your Mac with a free app. Matita bundles whisper.cpp with Metal acceleration, plus NVIDIA Parakeet, speaker names and summaries.",
     h1="Whisper on your Mac, without the Terminal",
     lead="Matita is a free app that runs Whisper large-v3-turbo on your Mac's GPU. Download it, drag it to Applications, drop in a recording.",
     crumb=[("Matita", ""), ("Whisper for Mac", "whisper-mac-app/")],
     related=[("Transcribe iPhone Voice Memos on your Mac", "guides/transcribe-voice-memos-on-mac/"),
              ("Otter alternative", "otter-alternative/"), ("Granola alternative", "granola-alternative/")],
     body=f"""
<p>Whisper is OpenAI's open speech recognition model. Running it yourself usually means Python, Homebrew and the command line. Matita packages <a href="https://github.com/ggml-org/whisper.cpp">whisper.cpp</a>, FFmpeg and everything else inside one Mac app, built for Apple Silicon with Metal.</p>
<h2>The speech models</h2>
<div class="tbl"><table><thead><tr><th>In Matita</th><th>Model</th><th>Download</th><th>Languages</th></tr></thead><tbody>
<tr><th>Turbo · Recommended</th><td>Whisper large-v3-turbo</td><td>1.6 GB</td><td>All the languages in Matita</td></tr>
<tr><th>Turbo 8-bit · Fast</th><td>Whisper large-v3-turbo, 8-bit</td><td>870 MB</td><td>All the languages in Matita</td></tr>
<tr><th>Parakeet · Fastest</th><td>NVIDIA Parakeet TDT 0.6B v3</td><td>640 MB</td><td>English, French, Spanish, German</td></tr>
</tbody></table></div>
<p>You can choose Korean, English, Japanese, Chinese, French, Spanish or German, or let Whisper detect the language. Matita downloads a model the first time you use it, then works offline.</p>
<h2>What Matita adds around Whisper</h2>
<ul>
<li>Speaker names, with a Speakers view that gathers each person's lines</li>
<li>Summaries and title ideas with Ollama on your Mac, or with Claude</li>
<li>Recording from your microphone, and from Zoom, Teams or Meet on macOS 14.2 or later</li>
<li>iPhone Voice Memos and watch folders, picked up automatically</li>
<li>Timestamps you can click to play, and words the model wasn't sure about highlighted</li>
<li>Export to .txt, .srt, Word and Markdown, and notes for Obsidian</li>
</ul>
{fig("time", "A transcript in Matita with timestamps and an audio player", "Every line has a timestamp. Click one to hear that part.", 1)}
<h2>Requirements</h2>
<p>A Mac with Apple Silicon (M1 or later) and macOS 14 Sonoma or later. About 100 MB for the app, plus the model you choose.</p>
""",
     faq=[("Is this the official Whisper app?", "No. Matita is an independent open-source app that runs the openly released Whisper models through whisper.cpp."),
          ("Does it use the Mac's GPU?", "Yes. whisper.cpp in Matita is built with Metal, so it runs on Apple Silicon's GPU."),
          ("Can I use it on an Intel Mac?", "Not this version. Matita needs Apple Silicon.")])

# ------------------------------------------------------------------ guides
page("guides/transcribe-voice-memos-on-mac/", lang="en",
     title="How to transcribe iPhone Voice Memos on your Mac, free and offline · Matita",
     desc="Step by step: transcribe iPhone Voice Memos on your Mac with Matita, and have new memos transcribed automatically. Free, offline, no upload.",
     h1="Transcribe iPhone Voice Memos on your Mac",
     lead="Record on your iPhone, and find the transcript waiting on your Mac. Free, offline, and nothing is uploaded.",
     crumb=[("Matita", ""), ("Guides", "guides/"), ("Voice Memos", "guides/transcribe-voice-memos-on-mac/")],
     related=[("Record a Zoom call without a bot", "guides/record-zoom-without-a-bot/"), ("Whisper for Mac", "whisper-mac-app/")],
     alt="ko/guides/voice-memos-to-text/",
     body="""
<h2>Before you start</h2>
<ul>
<li>A Mac with Apple Silicon and macOS 14 or later, with <a href="../../">Matita</a> installed.</li>
<li>Voice Memos syncing through iCloud. On your iPhone: Settings → your name → iCloud → turn on Voice Memos. Do the same on your Mac, then open the Voice Memos app on the Mac once so your recordings come across.</li>
</ul>
<h2>Steps</h2>
<ol class="steps">
<li><b>Click “From your iPhone”</b> on Matita's main window.</li>
<li><b>Give Matita Full Disk Access.</b> macOS keeps Voice Memos private, so the first time Matita shows you how: open System Settings → Privacy &amp; Security → Full Disk Access and turn on Matita (or add it with +). Then quit Matita (⌘Q) and open it again. Matita only reads your recordings; it never changes or uploads them.</li>
<li><b>Pick the memos</b> you want transcribed, or search them by title.</li>
<li><b>Turn on “Transcribe new memos automatically”</b> if you want every new memo done for you. Matita checks for new ones every couple of minutes, using the language and model on the main window.</li>
</ol>
<div class="box"><p><b>A memo is greyed out?</b> It's still in iCloud. Play it once in the Voice Memos app on your Mac to download it.</p></div>
<h2>Good settings for voice memos</h2>
<ul>
<li><b>Language:</b> choose yours, or Detect automatically if you mix languages between memos.</li>
<li><b>Model:</b> Turbo for the best accuracy. Parakeet is fastest for English, French, Spanish and German.</li>
<li><b>Titles:</b> with AI assist on, each transcript comes with title ideas, so “New Recording 23” can get a name you'll find later.</li>
</ul>
""",
     faq=[("Does Matita upload my voice memos?", "No. It reads them from your Mac's Voice Memos library and transcribes them on your Mac."),
          ("Why does Matita need Full Disk Access?", "macOS stores Voice Memos in a protected folder. Full Disk Access is the only way for another app to read it.")])

page("guides/record-zoom-without-a-bot/", lang="en",
     title="How to record and transcribe a Zoom call without a bot on Mac · Matita",
     desc="Record both sides of a Zoom, Teams or Google Meet call on your Mac and get a transcript with speaker names, without a meeting bot. Free with Matita.",
     h1="Record and transcribe a Zoom call without a bot",
     lead="Matita records your microphone and the other people in the call straight from your Mac. Nobody new joins the meeting.",
     crumb=[("Matita", ""), ("Guides", "guides/"), ("Zoom without a bot", "guides/record-zoom-without-a-bot/")],
     related=[("Granola alternative", "granola-alternative/"), ("Otter alternative", "otter-alternative/"),
              ("Transcribe iPhone Voice Memos on your Mac", "guides/transcribe-voice-memos-on-mac/")],
     body="""
<div class="box"><p><b>Ask first.</b> Tell the people in the call that you're recording. In many places recording a conversation without consent is against the law.</p></div>
<h2>Before you start</h2>
<ul>
<li>macOS 14.2 or later. Recording the sound of other apps needs it.</li>
<li>On Matita's main window, set <b>Speakers</b> to the number of people in the call (2 people, 3 people…). It separates voices better than Auto.</li>
</ul>
<h2>Steps</h2>
<ol class="steps">
<li><b>Click Record</b> just before the meeting starts.</li>
<li><b>Turn on “Include sound from other apps”.</b> This records the other people too. The first time, macOS asks you to allow Matita to use the microphone and to record audio from other apps.</li>
<li><b>Keep the recording window open.</b> Your Mac won't go to sleep while recording. Use Pause if you need a break.</li>
<li><b>Click “Stop and transcribe”</b> when the call ends.</li>
<li><b>Open the transcript.</b> Name the speakers by clicking “Speaker 1” and so on, and use the Speakers tab to read what each person said.</li>
</ol>
""" + fig("speakers", "Matita's Speakers view after a meeting", "The Speakers tab: each person's lines, with their share of the talking.", 2) + """
<h2>Afterwards</h2>
<ul>
<li>With AI assist on, you get an overview, action items and title ideas.</li>
<li>Save it as Word or Markdown, or send it to Obsidian.</li>
</ul>
""",
     faq=[("Does this work with Teams and Google Meet?", "Yes. Matita records the sound your Mac plays, so it works with any call app, including in a browser."),
          ("Does anyone see that I'm recording?", "No bot joins, so the call app doesn't announce it. That's why you should tell people yourself.")])

# ------------------------------------------------------------------ Korean guide
page("ko/guides/voice-memos-to-text/", lang="ko",
     title="맥에서 아이폰 음성메모·강의 녹음을 텍스트로 변환하는 법 (무료, 오프라인) · Matita",
     desc="아이폰 음성메모와 강의·회의 녹음을 Mac에서 무료로 텍스트로 바꾸는 방법. 인터넷 업로드 없이 Mac 안에서 받아쓰기하고 요약까지 만들어요.",
     h1="아이폰 음성메모와 강의 녹음을 Mac에서 텍스트로",
     lead="Matita를 쓰면 녹음 파일을 업로드하지 않고 Mac 안에서 바로 받아쓸 수 있어요. 무료이고 계정도 필요 없어요.",
     crumb=[("Matita", "ko/"), ("가이드", "ko/guides/"), ("음성메모 텍스트 변환", "ko/guides/voice-memos-to-text/")],
     related=[("Matita 소개", "ko/"), ("Whisper for Mac (영문)", "whisper-mac-app/")],
     alt="guides/transcribe-voice-memos-on-mac/",
     body="""
<h2>준비</h2>
<ul>
<li>Apple Silicon(M1 이상) Mac, macOS 14 이상, <a href="../../">Matita</a> 설치</li>
<li>아이폰과 Mac 모두 iCloud에서 음성 메모를 켜 주세요. 아이폰: 설정 → 내 이름 → iCloud → 음성 메모. 그다음 Mac에서 음성 메모 앱을 한 번 열면 녹음이 동기화돼요.</li>
<li>첫 화면에서 <b>Language</b>를 Korean으로 고르세요. 한국어는 Turbo 모델을 쓰세요. Parakeet은 영어·프랑스어·스페인어·독일어 전용이에요.</li>
</ul>
<h2>아이폰 음성메모 받아쓰기</h2>
<ol class="steps">
<li><b>“From your iPhone”</b>을 눌러요.</li>
<li><b>전체 디스크 접근 권한을 줘요.</b> macOS가 음성 메모를 보호하고 있어서 처음 한 번 필요해요. 시스템 설정 → 개인정보 보호 및 보안 → 전체 디스크 접근 권한에서 Matita를 켜고(목록에 없으면 +로 추가), Matita를 종료(⌘Q)했다가 다시 열어요. Matita는 녹음을 읽기만 하고, 바꾸거나 업로드하지 않아요.</li>
<li><b>받아쓸 메모를 골라요.</b> 제목으로 검색할 수도 있어요.</li>
<li><b>“Transcribe new memos automatically”</b>를 켜면 새 메모가 생길 때마다 자동으로 받아써요.</li>
</ol>
<div class="box"><p><b>메모가 회색으로 보이면</b> 아직 iCloud에만 있는 거예요. Mac의 음성 메모 앱에서 한 번 재생하면 내려받아져요.</p></div>
<h2>강의·회의를 바로 녹음하기</h2>
<ol class="steps">
<li><b>Record</b>를 눌러 녹음을 시작해요. 녹음하는 동안 Mac이 잠자기에 들어가지 않아요.</li>
<li>Zoom·Teams 수업이나 회의라면 <b>“Include sound from other apps”</b>를 켜요. 상대방 목소리까지 함께 녹음돼요(macOS 14.2 이상).</li>
<li><b>“Stop and transcribe”</b>를 누르면 받아쓰기가 시작돼요.</li>
</ol>
<div class="box"><p>다른 사람이 있는 강의나 회의를 녹음할 때는 미리 동의를 구하세요.</p></div>
<h2>이미 있는 녹음 파일</h2>
<p>m4a, mp3, wav, mp4 같은 파일을 창에 끌어다 놓거나 ⌘O로 열면 돼요. 녹음기 SD 카드나 Zoom 녹화 폴더를 <b>Watch folders</b>에 추가해 두면 새 파일이 생길 때 자동으로 받아써요.</p>
<h2>받아쓴 다음</h2>
<ul>
<li>여러 사람이 말한 녹음이라면 받아쓰기 전에 <b>Speakers</b>를 실제 인원수로 정해 두세요. 끝나면 Speakers 탭에서 사람별로 모아 볼 수 있어요.</li>
<li>AI 요약을 켜면 개요, 할 일, 제목 후보를 만들어 줘요. Ollama를 쓰면 요약도 Mac 안에서 만들어져요.</li>
<li>Word, Markdown, 자막(.srt)으로 저장하거나 Obsidian 노트로 보낼 수 있어요.</li>
</ul>
""",
     faq=[("정말 무료인가요?", "네. MIT 라이선스의 오픈소스 앱이고, 유료 플랜이나 사용량 제한, 계정이 없어요."),
          ("인터넷이 필요한가요?", "처음에 음성 모델을 내려받을 때와 업데이트 확인, 요약에 Claude를 고른 경우에만 인터넷을 써요. 받아쓰기는 오프라인에서 돼요."),
          ("Windows나 인텔 Mac에서도 되나요?", "지금은 Apple Silicon(M1 이상) Mac만 지원해요.")])


# ------------------------------------------------------------------ guide indexes
def index_page(path, lang, title, desc, h1, lead, items, crumb, alt):
    up = "../" * path.count("/")
    li = "".join('<li><a href="%s%s"><b>%s</b></a><br><span class="muted">%s</span></li>' % (up, p, html.escape(t), html.escape(d))
                 for t, d, p in items)
    page(path, lang=lang, title=title, desc=desc, h1=h1, lead=lead, crumb=crumb, alt=alt,
         related=[("Matita", "ko/" if lang == "ko" else "")],
         body='<ul class="list">%s</ul>' % li)


index_page("guides/", "en", "Guides · Matita", "How-to guides for Matita, the free offline transcription app for Mac.",
           "Guides", "Short, step-by-step guides for getting the most out of Matita.",
           [("Transcribe iPhone Voice Memos on your Mac", "Record on your iPhone, read it on your Mac.", "guides/transcribe-voice-memos-on-mac/"),
            ("Record and transcribe a Zoom call without a bot", "Both sides of Zoom, Teams or Meet, with speaker names.", "guides/record-zoom-without-a-bot/"),
            ("Whisper on your Mac, without the Terminal", "The models in Matita and what they're good at.", "whisper-mac-app/"),
            ("A free, offline Otter alternative", "How Matita and Otter differ.", "otter-alternative/"),
            ("A free Granola alternative", "How Matita and Granola differ.", "granola-alternative/")],
           [("Matita", ""), ("Guides", "guides/")], "ko/guides/")
index_page("ko/guides/", "ko", "가이드 · Matita", "무료 오프라인 Mac 받아쓰기 앱 Matita 사용 가이드.",
           "가이드", "Matita를 더 잘 쓰는 방법을 단계별로 정리했어요.",
           [("아이폰 음성메모와 강의 녹음을 Mac에서 텍스트로", "음성메모 자동 받아쓰기, 강의·회의 녹음, 기존 파일까지.", "ko/guides/voice-memos-to-text/"),
            ("Record a Zoom call without a bot (영문)", "봇 없이 Zoom·Teams·Meet 회의 녹음하기.", "guides/record-zoom-without-a-bot/"),
            ("Whisper for Mac (영문)", "Matita에 들어 있는 음성 모델 소개.", "whisper-mac-app/")],
           [("Matita", "ko/"), ("가이드", "ko/guides/")], "guides/")

# ------------------------------------------------------------------ sitemap
urls = ["", "ko/"] + [p for p, _ in PAGES]
with open(os.path.join(ROOT, "sitemap.xml"), "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
    for u in urls:
        f.write("  <url><loc>%s%s</loc></url>\n" % (BASE, u))
    f.write("</urlset>\n")
print("wrote %d pages and sitemap.xml" % len(PAGES))
