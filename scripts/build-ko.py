#!/usr/bin/env python3
"""Builds docs/ko/index.html from docs/index.html: same page, Korean text, links one folder up.

    python3 scripts/build-ko.py

Every English string below must still be on the English page; the script stops if one isn't,
so a change to docs/index.html can't silently leave English text on the Korean page.
"""
import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
BASE = "https://joonkiee.github.io/matita-transcriber/"
s = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()

s = re.sub(r'<script type="application/ld\+json">.*?</script>', lambda m: '<script type="application/ld+json">%s</script>' % json.dumps([{
    "@context": "https://schema.org", "@type": "SoftwareApplication", "name": "Matita", "inLanguage": "ko",
    "description": "Mac과 Windows용 무료 오픈소스 받아쓰기 앱. 회의, 강의, 통화, 녹음 파일을 내 컴퓨터 안에서 Whisper로 텍스트로 바꾸고, 화자 구분, AI 요약, Obsidian 노트까지 만들어요.",
    "applicationCategory": "BusinessApplication", "operatingSystem": "macOS 14 이상 (Apple Silicon), Windows 10·11 (64비트)",
    "url": BASE + "ko/", "downloadUrl": "https://github.com/joonkiee/matita-transcriber/releases/latest", "image": BASE + "og.png",
    "license": "https://opensource.org/licenses/MIT", "isAccessibleForFree": True,
    "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
    "author": {"@type": "Person", "name": "Jun Ki Shin", "url": "https://github.com/joonkiee"}}], ensure_ascii=False), s, flags=re.S)

T = [
    ('<html lang="en">', '<html lang="ko">'),
    ('<title>Matita · Free offline transcription app for Mac and Windows</title>', '<title>Matita · 무료 녹음 텍스트 변환 앱 (Mac·Windows, 오프라인 받아쓰기)</title>'),
    ('content="Free, open-source transcription app for Mac and Windows. Turn meetings, lectures, calls and recordings into transcripts, summaries and notes, offline with Whisper. No account."',
     'content="Mac과 Windows용 무료 받아쓰기 앱. 회의·강의·통화 녹음을 내 컴퓨터 안에서 텍스트로 바꾸고 요약까지 만들어요. 업로드 없이, 계정 없이, 한국어 지원."'),
    ('<link rel="canonical" href="%s">' % BASE, '<link rel="canonical" href="%sko/">' % BASE),
    ('<meta property="og:url" content="%s">' % BASE, '<meta property="og:url" content="%sko/">\n<meta property="og:locale" content="ko_KR">' % BASE),
    ('<meta property="og:title" content="Matita · Free offline transcription app for Mac and Windows">', '<meta property="og:title" content="Matita · 무료 녹음 텍스트 변환 앱 (Mac·Windows)">'),
    ('<meta property="og:description" content="Transcripts, summaries and notes from your recordings, made on your own computer. Free and open source.">',
     '<meta property="og:description" content="녹음을 내 컴퓨터 안에서 텍스트, 요약, 노트로. 무료 오픈소스, 계정 필요 없음.">'),
    # nav
    ('<a href="#features">Features</a><a href="#compare">Compare</a><a href="#install">Install</a><a href="#faq">FAQ</a><a href="guides/">Guides</a><a href="ko/" lang="ko">한국어</a>',
     '<a href="#features">기능</a><a href="#compare">비교</a><a href="#install">설치</a><a href="#faq">FAQ</a><a href="ko/guides/">가이드</a><a href="./" lang="en">English</a>'),
    ('>Download</a><a class="btn plain"', '>다운로드</a><a class="btn plain"'),
    # hero
    ('<h1>Your recordings <span', '<h1>녹음을 <span'),
    ('</span><br>written down <span', '</span><br>글로 바꿔요 <span'),
    ('<p class="hsub">Transcripts, summaries and notes from meetings, lectures and calls, made on your own computer. Free, open source, and no account.</p>',
     '<p class="hsub">회의, 강의, 통화를 내 컴퓨터 안에서 텍스트와 요약, 노트로 바꿔요. 무료 오픈소스이고 계정도 필요 없어요.</p>'),
    ('<span class="dl-label">Download</span>', '<span class="dl-label">다운로드</span>'),
    ('<span class="ver">Free and open source</span><span>Mac: Apple Silicon, macOS 14+</span><span>Windows 10 and 11 (beta)</span>',
     '<span class="ver">무료 · 오픈소스</span><span>Mac: Apple Silicon, macOS 14 이상</span><span>Windows 10·11 (베타)</span>'),
    ('alt="Matita: drop area, settings and the list of transcripts"', 'alt="Matita 메인 화면: 파일 추가, 설정, 받아쓴 목록"'),
    # why
    ("<h3>Stays on your computer</h3><p>Speech recognition runs on your own machine. Your audio isn't uploaded.</p>",
     '<h3>컴퓨터 밖으로 나가지 않아요</h3><p>음성 인식이 내 컴퓨터에서 돌아가요. 녹음 파일을 업로드하지 않아요.</p>'),
    ('<h3>Free, no limits</h3><p>No subscription, no minute caps, no account to make.</p>', '<h3>무료, 제한 없음</h3><p>구독도, 분 단위 제한도, 가입도 없어요.</p>'),
    ('<h3>Plain files you keep</h3><p>Text, subtitles, Word and Markdown, saved in a folder you choose.</p>', '<h3>내가 갖는 파일</h3><p>텍스트, 자막, Word, Markdown을 내가 고른 폴더에 저장해요.</p>'),
    ('>Why Matita</span>', '>왜 Matita인가요</span>'),
    ("<h2 style=\"margin-top:16px\">A transcription app that works <mark>on your computer</mark>, not on someone else's server.</h2>",
     '<h2 style="margin-top:16px">남의 서버가 아니라 <mark>내 컴퓨터에서</mark> 돌아가는 받아쓰기 앱.</h2>'),
    # features
    ('<span class="tag">Features</span><h2>Every kind of<br>recording <span', '<span class="tag">기능</span><h2>어떤 녹음이든<br><span'),
    ('</span> covered</h2>', '</span> 다 받아써요</h2>'),
    ('<p class="aside">Korean, English, Japanese, Chinese, French, Spanish and German, or let Whisper detect the language. Parakeet is the fast option for English, French, Spanish and German.</p>',
     '<p class="aside">한국어, 영어, 일본어, 중국어, 프랑스어, 스페인어, 독일어를 고르거나 자동 감지에 맡길 수 있어요. 영어·프랑스어·스페인어·독일어는 더 빠른 Parakeet도 쓸 수 있어요.</p>'),
    ('<h3>Meetings and calls</h3>\n          <p>On a Mac, records you and the other people in Zoom, Teams or Meet. Then labels who said what. No bot joins the call.</p>',
     '<h3>회의와 통화</h3>\n          <p>Mac에서는 Zoom, Teams, Meet의 내 목소리와 상대방 목소리를 함께 녹음해요. 누가 말했는지도 구분해요. 회의에 봇이 들어가지 않아요.</p>'),
    ('<span>Speaker names</span>', '<span>화자 구분</span>'),
    ('<h3>Summaries and notes</h3>\n          <p>An overview, action items and three title ideas for every recording. Made on your computer with Ollama, or with Claude if you choose. Notes can go straight into Obsidian.</p>',
     '<h3>요약과 노트</h3>\n          <p>녹음마다 개요, 할 일, 제목 후보 세 개를 만들어요. Ollama로 내 컴퓨터 안에서 만들거나, 원하면 Claude를 쓸 수 있어요. Obsidian 노트로 바로 보낼 수도 있어요.</p>'),
    ("<h3>Lectures and voice memos</h3>\n          <p>Record with your computer's microphone. On a Mac, iPhone Voice Memos are transcribed as soon as they arrive.</p>",
     '<h3>강의와 음성메모</h3>\n          <p>컴퓨터 마이크로 녹음하세요. Mac에서는 아이폰 음성 메모가 동기화되는 대로 자동으로 받아써요.</p>'),
    ('<span>Microphone</span><span>Voice Memos</span>', '<span>마이크</span><span>음성 메모</span>'),
    ("<h3>Recordings you already have</h3>\n          <p>Drop in any audio or video file, or point Matita at a folder, such as a recorder's SD card, and new files are picked up.</p>",
     '<h3>이미 있는 녹음 파일</h3>\n          <p>오디오나 영상 파일을 끌어다 놓으세요. 녹음기 SD 카드 같은 폴더를 지정해 두면 새 파일을 알아서 받아써요.</p>'),
    ('<span>Watch folders</span>', '<span>감시 폴더</span>'),
    ('alt="An AI summary with three title ideas"', 'alt="제목 후보 세 개가 있는 AI 요약"'),
    ('<figcaption><b>Read instead of replaying.</b> The summary is ready when the transcript is.</figcaption>', '<figcaption><b>다시 듣지 말고 읽으세요.</b> 받아쓰기가 끝나면 요약도 준비돼요.</figcaption>'),
    ('alt="A transcript with timestamps and an audio player"', 'alt="시간이 표시된 받아쓰기와 오디오 플레이어"'),
    ('<figcaption><b>Find the moment.</b> Every line has a timestamp. Click it to hear that part.</figcaption>', '<figcaption><b>그 순간을 바로 찾기.</b> 모든 줄에 시간이 있어요. 누르면 그 부분이 재생돼요.</figcaption>'),
    # compare
    ('<span class="tag">Compare</span><h2>How Matita compares</h2>', '<span class="tag">비교</span><h2>다른 서비스와 비교</h2>'),
    ('<p class="aside">Otter, Granola and Wispr Flow are cloud services. Here is what that changes.</p>', '<p class="aside">Otter, Granola, Wispr Flow는 클라우드 서비스예요. 그래서 이런 차이가 생겨요.</p>'),
    ('aria-label="Compare Matita with"', 'aria-label="비교할 서비스"'),
    ("<tr><th>Where speech becomes text</th><td class=\"mc\">On your computer</td><td class=\"ot\">Otter's cloud</td><td class=\"gr\">A cloud transcription provider</td><td class=\"wf\">Wispr's cloud</td></tr>",
     '<tr><th>음성 인식 위치</th><td class="mc">내 컴퓨터</td><td class="ot">Otter 클라우드</td><td class="gr">외부 클라우드 전사 업체</td><td class="wf">Wispr 클라우드</td></tr>'),
    ('<tr><th>Files you already have</th><td class="mc">Any audio or video file</td><td class="ot">3 files on the free plan</td><td class="gr">Not supported</td><td class="wf">Not supported</td></tr>',
     '<tr><th>기존 녹음 파일</th><td class="mc">모든 오디오·영상 파일</td><td class="ot">무료 플랜은 3개까지</td><td class="gr">지원 안 함</td><td class="wf">지원 안 함</td></tr>'),
    ('<tr><th>Free plan</th><td class="mc">Everything, no limits</td><td class="ot">300 minutes a month, 30 minutes per conversation</td><td class="gr">Meeting history for 30 days</td><td class="wf">2,000 words of dictation a week</td></tr>',
     '<tr><th>무료 플랜</th><td class="mc">모든 기능, 제한 없음</td><td class="ot">월 300분, 대화당 30분</td><td class="gr">회의 기록 30일 보관</td><td class="wf">주 2,000단어 받아쓰기</td></tr>'),
    ('<tr><th>Paid plan</th><td class="mc">None</td><td class="ot">$16.99 a month, or $8.33 a month billed yearly</td><td class="gr">$14 per user a month</td><td class="wf">$15 a month, or $12 a month billed yearly</td></tr>',
     '<tr><th>유료 플랜</th><td class="mc">없음</td><td class="ot">월 $16.99, 연간 결제 시 월 $8.33</td><td class="gr">사용자당 월 $14</td><td class="wf">월 $15, 연간 결제 시 월 $12</td></tr>'),
    ('<tr><th>Account</th><td class="mc">Not needed</td><td class="ot">Required</td><td class="gr">Required</td><td class="wf">Required</td></tr>',
     '<tr><th>계정</th><td class="mc">필요 없음</td><td class="ot">필요</td><td class="gr">필요</td><td class="wf">필요</td></tr>'),
    ('<p class="note">Checked on 5 November 2026 against ', '<p class="note">2026년 11월 5일 각 사 공식 페이지 기준으로 확인했어요: '),
    ('. Prices in US dollars, per user, before tax. They may have changed since.</p>', '. 가격은 미국 달러, 사용자 1인, 세금 제외 기준이고 이후 바뀌었을 수 있어요.</p>'),
    # install
    ('<span class="tag">Install</span><h2>Running in a minute</h2>', '<span class="tag">설치</span><h2>1분이면 시작해요</h2>'),
    ("<p class=\"aside\">Matita isn't signed with a paid certificate, so your computer asks you to confirm once.</p>",
     '<p class="aside">Matita는 유료 인증서로 서명하지 않아서, 처음 한 번 컴퓨터가 확인을 요청해요.</p>'),
    ('aria-label="Your computer"', 'aria-label="내 컴퓨터"'),
    ('data-os="win" aria-selected="false">Windows <small>beta</small>', 'data-os="win" aria-selected="false">Windows <small>베타</small>'),
    ('<h3>Download and drag</h3><p>Open the .dmg and drag Matita into Applications.</p>', '<h3>받아서 끌어 놓기</h3><p>.dmg를 열고 Matita를 응용 프로그램 폴더로 끌어다 놓아요.</p>'),
    ("<h3>Open it once</h3><p>macOS says it can't check the app. Click <b>Done</b>.</p>", '<h3>한 번 열기</h3><p>macOS가 앱을 확인할 수 없다고 하면 <b>완료</b>를 눌러요.</p>'),
    ('<h3>Open Anyway</h3><p>System Settings → Privacy &amp; Security, then <b>Open Anyway</b>. You only do this once.</p>',
     '<h3>그래도 열기</h3><p>시스템 설정 → 개인정보 보호 및 보안에서 <b>그래도 열기</b>를 눌러요. 처음 한 번만 하면 돼요.</p>'),
    ('<p class="tip" data-os="mac">Prefer Terminal? ', '<p class="tip" data-os="mac">터미널이 편하다면: '),
    ('<h3>Run the installer</h3><p>It installs for your account only. No administrator password needed.</p>', '<h3>설치 파일 실행</h3><p>내 계정에만 설치돼요. 관리자 암호가 필요 없어요.</p>'),
    ('<h3>If Windows warns you</h3><p>On “Windows protected your PC”, click <b>More info</b>, then <b>Run anyway</b>.</p>',
     '<h3>Windows가 경고하면</h3><p>“Windows의 PC 보호” 창에서 <b>추가 정보</b>를 누르고 <b>실행</b>을 눌러요.</p>'),
    ('<h3>Open Matita</h3><p>From the Start menu. It opens in its own window; closing the window quits Matita.</p>',
     '<h3>Matita 열기</h3><p>시작 메뉴에서 열어요. 별도 창으로 열리고, 창을 닫으면 Matita도 종료돼요.</p>'),
    ('<p class="tip" data-os="win" hidden>Needs 64-bit Windows 10 or 11. Matita uses your graphics card through Vulkan when its driver supports it, and the processor otherwise (slower).</p>',
     '<p class="tip" data-os="win" hidden>64비트 Windows 10 또는 11이 필요해요. 그래픽카드 드라이버가 Vulkan을 지원하면 그래픽카드로, 아니면 CPU로 받아써요(더 느려요).</p>'),
    # faq
    ('<h2>Questions</h2>', '<h2>자주 묻는 질문</h2>'),
    ("<details><summary>Does anything leave my computer?</summary><p>Your audio and transcripts don't. Matita goes online to download a speech model the first time, to check GitHub for a new version, and, only if you choose Claude for summaries, to send the transcript to Anthropic.</p></details>",
     '<details><summary>컴퓨터 밖으로 나가는 게 있나요?</summary><p>녹음과 받아쓴 글은 나가지 않아요. 처음 음성 모델을 내려받을 때, GitHub에서 새 버전을 확인할 때, 그리고 요약에 Claude를 고른 경우 글을 Anthropic에 보낼 때만 인터넷을 써요.</p></details>'),
    ('<details><summary>How much space does it need?</summary><p>About 100 MB for the app, plus one speech model on first use: 640 MB (Parakeet), 870 MB (Turbo 8-bit) or 1.6 GB (Turbo).</p></details>',
     '<details><summary>용량은 얼마나 필요한가요?</summary><p>앱은 약 100 MB이고, 처음 쓸 때 음성 모델 하나를 내려받아요: 640 MB(Parakeet), 870 MB(Turbo 8-bit), 1.6 GB(Turbo). 한국어는 Turbo를 쓰세요.</p></details>'),
    ("<details><summary>Which computers does it run on?</summary><p>Macs with Apple Silicon (M1 or later) on macOS 14 Sonoma or later, and 64-bit Windows 10 and 11 (beta). Intel Macs aren't supported. Recording the sound of other apps, for calls, needs macOS 14.2; it and iPhone Voice Memos are Mac-only for now.</p></details>",
     '<details><summary>어떤 컴퓨터에서 되나요?</summary><p>Apple Silicon(M1 이상) Mac과 macOS 14 Sonoma 이상, 그리고 64비트 Windows 10·11(베타)에서 돼요. 인텔 Mac은 지원하지 않아요. 통화 상대방 목소리 녹음(macOS 14.2 이상 필요)과 아이폰 음성 메모는 지금은 Mac에서만 돼요.</p></details>'),
    ("<details><summary>Why do macOS and Windows warn me the first time?</summary><p>They show that warning for any app that isn't signed with a paid certificate. Matita's source is public, and each version is built from it on GitHub.</p></details>",
     '<details><summary>처음 열 때 왜 경고가 뜨나요?</summary><p>유료 인증서로 서명하지 않은 앱이면 macOS와 Windows가 항상 이 경고를 보여줘요. Matita는 소스 코드가 공개되어 있고, 모든 버전을 GitHub에서 그 코드로 빌드해요.</p></details>'),
    ('<details><summary>How do I remove it?</summary><p>Mac: quit Matita and drag it to the Trash; models are in ~/whisper-models and settings in ~/whisper-app. Windows: Settings → Apps → Matita → Uninstall; models and settings are in %LOCALAPPDATA%\\Matita. Your transcripts stay in your Transcripts folder.</p></details>',
     '<details><summary>삭제는 어떻게 하나요?</summary><p>Mac: Matita를 종료하고 휴지통으로 끌어다 놓으세요. 모델은 ~/whisper-models, 설정은 ~/whisper-app에 있어요. Windows: 설정 → 앱 → Matita → 제거. 모델과 설정은 %LOCALAPPDATA%\\Matita에 있어요. 받아쓴 글은 Transcripts 폴더에 그대로 남아요.</p></details>'),
    ('<details><summary>How can I help?</summary><p>Report problems or share ideas on <a href="https://github.com/joonkiee/matita-transcriber/issues/new/choose">GitHub</a>, or give the project a star.</p></details>',
     '<details><summary>어떻게 도울 수 있나요?</summary><p><a href="https://github.com/joonkiee/matita-transcriber/issues/new/choose">GitHub</a>에 문제나 아이디어를 남기거나 별(Star)을 눌러 주세요.</p></details>'),
    # end
    ('<h2>Try it on your next recording</h2>\n  <p>Free and open source. No account.</p>', '<h2>다음 녹음에 써 보세요</h2>\n  <p>무료 오픈소스, 계정 필요 없음.</p>'),
    ('>View on GitHub</a>', '>GitHub에서 보기</a>'),
    # footer
    ('<span class="more"><a href="guides/">Guides</a> · <a href="otter-alternative/">Otter alternative</a> · <a href="granola-alternative/">Granola alternative</a> · <a href="whisper-mac-app/">Whisper for Mac</a> · <a href="ko/" lang="ko">한국어</a></span>',
     '<span class="more"><a href="ko/guides/">가이드</a> · <a href="ko/guides/voice-memos-to-text/">음성메모 텍스트 변환</a> · <a href="otter-alternative/">Otter 대안 (영문)</a> · <a href="granola-alternative/">Granola 대안 (영문)</a> · <a href="./" lang="en">English</a></span>'),
    ('<span>Matita · MIT licence · Built with whisper.cpp, sherpa-onnx and FFmpeg</span>', '<span>Matita · MIT 라이선스 · whisper.cpp, sherpa-onnx, FFmpeg 사용</span>'),
    ('>Report a problem</a>', '>문제 신고</a>'),
    # dialogs
    ('<h3 id="afterTitle">Your download has started</h3>\n  <p>Then:</p>', '<h3 id="afterTitle">다운로드를 시작했어요</h3>\n  <p>그다음:</p>'),
    ('<li>Open <b class="afterFile">the .dmg</b> and drag Matita into Applications.</li>', '<li><b class="afterFile">.dmg</b>를 열고 Matita를 응용 프로그램 폴더로 끌어다 놓아요.</li>'),
    ("<li>Open Matita. When macOS says it can't check it, click <b>Done</b>.</li>", '<li>Matita를 열어요. 확인할 수 없다는 창이 뜨면 <b>완료</b>를 눌러요.</li>'),
    ('<li>System Settings → Privacy &amp; Security → <b>Open Anyway</b>.</li>', '<li>시스템 설정 → 개인정보 보호 및 보안 → <b>그래도 열기</b>.</li>'),
    ('<li>Run <b class="afterFile">the installer</b> from your Downloads.</li>', '<li>다운로드 폴더의 <b class="afterFile">설치 파일</b>을 실행해요.</li>'),
    ('<li>If Windows says it protected your PC, click <b>More info</b>, then <b>Run anyway</b>.</li>', '<li>“Windows의 PC 보호” 창이 뜨면 <b>추가 정보</b> → <b>실행</b>을 눌러요.</li>'),
    ('<li>Matita opens in its own window. Next time, find it in the Start menu.</li>', '<li>Matita가 별도 창으로 열려요. 다음부터는 시작 메뉴에서 열면 돼요.</li>'),
    ('>Star on GitHub</a>', '>GitHub 별 누르기</a>'),
    ('data-close>Got it</button>', 'data-close>확인</button>'),
    ('<h3 id="notmacTitle">Matita is for Mac and Windows</h3>\n  <p>Send yourself the link and download it on your computer.</p>',
     '<h3 id="notmacTitle">Matita는 Mac과 Windows용이에요</h3>\n  <p>링크를 나에게 보내서 컴퓨터에서 받으세요.</p>'),
    ('id="copyLink">Copy link</button>', 'id="copyLink">링크 복사</button>'),
    ('id="mailLink" href="#">Email me the link</a>', 'id="mailLink" href="#">메일로 보내기</a>'),
    # script strings
    ('const page = "https://joonkiee.github.io/matita-transcriber/";', 'const page = "https://joonkiee.github.io/matita-transcriber/ko/";'),
    ('''  const T = {mac: "Download for Mac", win: "Download for Windows", other: "Get it for your computer",
             alsoWin: "Also for Windows (beta)", alsoMac: "Also for Mac", copied: "Copied", ver: "Version ",
             mail: ["Matita for Mac and Windows", "Download Matita on your computer: "]};''',
     '''  const T = {mac: "Mac용 다운로드", win: "Windows용 다운로드", other: "컴퓨터에서 받기",
             alsoWin: "Windows용도 있어요 (베타)", alsoMac: "Mac용도 있어요", copied: "복사했어요", ver: "버전 ",
             mail: ["Matita (Mac·Windows)", "컴퓨터에서 Matita 받기: "]};'''),
]
for a, b in T:
    if a not in s:
        raise SystemExit("not on the English page any more:\n  " + a[:120])
    s = s.replace(a, b)

# links and assets are one folder up
s = re.sub(r'((?:src|href|srcset)=")(?!https?:|#|mailto:|data:|\.\./)([^"]*)"',
           lambda m: m.group(1) + ("../" + m.group(2) if m.group(2) else "") + '"', s)
s = s.replace("url(fonts/", "url(../fonts/").replace('href="../ko/guides/', 'href="guides/').replace('href=".././"', 'href="../"')
s = s.replace("</style>", ":lang(ko) body{word-break:keep-all}\n:lang(ko) h1,:lang(ko) h2,:lang(ko) h3{letter-spacing:-.015em}\n</style>", 1)
os.makedirs(os.path.join(ROOT, "ko"), exist_ok=True)
with open(os.path.join(ROOT, "ko", "index.html"), "w", encoding="utf-8") as fh:
    fh.write(s)
print("wrote docs/ko/index.html")
