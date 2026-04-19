package com.qxotic.toknroll.testkit;

import java.util.List;

/** Centralized test string corpora for tokenizer and splitter tests. */
public final class TestCorpora {

    private TestCorpora() {}

    /**
     * Single source of truth for reusable tokenizer/splitter strings.
     *
     * <p>All other public corpora in this class are derived from this list so test suites share the
     * same text inputs.
     */
    public static final List<String> TOKENIZER_UNIVERSAL_SAMPLES =
            List.of(
                    "",
                    "plain ascii words",
                    "Hello, world!!!",
                    "It's we're THEY'LL I'd",
                    "don't I'M WE'RE THEY'LL",
                    "digits 0 1 12 123 1234 12345 123456",
                    "long digits 123456789012345678901234567890",
                    "numeric mix v1beta2 build42 x86_64 arm64",
                    "versions: v1 v2.5 v10.12.003 build-20260403",
                    "numbers with punctuation 3.14159 -42 +7e10 1,000,000",
                    "dates and times: 2026-04-03 23:59:59 03/04/2026",
                    "unicode digits ١٢٣ ۱۲۳ १२३ １２３",
                    "line1\rline2\r\nline3\nline4",
                    "tabs\t\tspaces  end",
                    "leading separators ,,,hello",
                    "trailing separators world!!!",
                    "adjacent separators !!!???...,,;;::",
                    "boundary punctuation (start) [token] (end)",
                    "path/to/api/v2.5.1?x=1&y=22",
                    "slashes and CRLF: a/b\\c\r\nnext",
                    "json: {\"a\":1,\"b\":[true,false,null]}",
                    "code: for(i=0;i<10;i++){sum+=i;} // done",
                    "a\u0000b\u0001c\u001Fd",
                    "emoji 😀 fallback",
                    "emoji 👨‍👩‍👧‍👦 and flags 🇺🇳🇯🇵",
                    "ZWJ family 👨‍👩‍👧‍👦 + kiss 👩‍❤️‍💋‍👨",
                    "skin tones 👍🏻👍🏽👍🏿",
                    "keycaps 1️⃣ 2️⃣ #️⃣ *️⃣",
                    "variation selectors ✊︎ ✊️",
                    "combining e\u0301 vs é, n\u0303 vs ñ",
                    "combining without base: \u0301\u0308\u0304",
                    "normalization pairs: é e\u0301 Å A\u030A Å",
                    "bidi marks A\u200EB\u200FC \u202Astart\u202C",
                    "directional isolates: A\u2066B\u2069 C\u2067D\u2069",
                    "word joiner + zwnbsp: a\u2060b\uFEFFc",
                    "zero-width controls: a\u200Bb\u200Cc\u200Dd\uFE0Fe",
                    "rtl عربي עברית فارسی",
                    "rtl mix العربية English עברית 123",
                    "arabic detailed: مرحبًا بالعالم، كيف الحال؟ الإصدار ٢٫٥ جاهز.",
                    "arabic with tatweel: العـــــربية",
                    "arabic harakat: العَرَبِيَّةُ",
                    "arabic numerals: ٠١٢٣٤٥٦٧٨٩",
                    "hebrew detailed: שלום עולם, מה שלומך? גרסה 2.5 מוכנה.",
                    "persian detailed: سلام دنیا، حالت چطوره؟ نسخه ۲٫۵ آماده است.",
                    "urdu detailed: ہیلو دنیا، آپ کیسے ہیں؟ ورژن ۲٫۵ تیار ہے۔",
                    "pashto detailed: سلام نړۍ، ته څنګه یې؟ نسخه ۲٫۵ چمتو ده.",
                    "rtl punctuation mix: العربية؛ עברית، فارسی؟ yes/no.",
                    "cjk 你好 日本語 한글",
                    "chinese simplified: 今天天气很好，我们一起去公园散步，然后喝茶。",
                    "chinese traditional: 今天天氣很好，我們一起去公園散步，然後喝茶。",
                    "japanese mixed: 今日は良い天気です。カタカナとひらがな、そして漢字。",
                    "japanese numbers: 第3版は2026年4月3日に公開されました。",
                    "thai ไทยภาษาไทย without spaces",
                    "thai detailed: สวัสดีชาวโลก วันนี้อากาศดีมาก ไปเดินเล่นกันไหม",
                    "thai with digits: เวอร์ชัน 2.5 เปิดตัววันที่ 03/04/2026",
                    "thai tone marks: ก่า ก้า ก๊า ก๋า",
                    "thai combining sample: นำ นํา",
                    "vietnamese: Xin chào thế giới, hôm nay trời đẹp quá!",
                    "vietnamese combining: tiếng Việt có dấu",
                    "vietnamese stacked accents: Trường hợp đặc biệt",
                    "indic नमस्ते दुनिया தமிழ் ಕನ್ನಡ বাংলা",
                    "turkish: Merhaba dünya, nasılsın? Sürüm 2.5 hazır.",
                    "russian: Привет, мир! Версия 2.5 уже доступна.",
                    "greek: Γειά σου κόσμε, η έκδοση 2.5 είναι έτοιμη.",
                    "hangul 한글 테스트",
                    "math ∑∏√∞ ≠ ≤ ≥",
                    "private use \uE000\uE001",
                    "noncharacters \uFDD0 \uFDEF",
                    "lone surrogates \uD800 \uDC00 end",
                    "punctuation []{}()!?.,;:/\\\"'",
                    "long punctuation !!!???...,,;;::--__",
                    "NBSP only: a\u00A0b\u00A0c",
                    "NEL only: a\u0085b\u0085c",
                    "figure space: a\u2007b\u2007c",
                    "narrow no-break space: a\u202Fb\u202Fc",
                    "ogham space mark: a\u1680b\u1680c",
                    "en/em quads: a\u2000b\u2001c",
                    "thin/hair spaces: a\u2009b\u200Ac",
                    "line separator: a\u2028b",
                    "paragraph separator: a\u2029b",
                    "ideographic space: a\u3000b",
                    "mixed unicode spaces: a\u00A0\u2007\u202F\u3000b",
                    "CR LF CRLF NEL LS PS: a\rb\nc\r\nd\u0085e\u2028f\u2029g",
                    "arabic with nbsp: مرحبًا\u00A0بالعالم",
                    "thai with nbsp: สวัสดี\u00A0ชาวโลก",
                    "vietnamese with nbsp: Xin\u00A0chào\u00A0thế\u00A0giới",
                    "emoji + unicode spaces: 😀\u00A0😀\u202F😀\u3000😀",
                    "spaces  \t\n\r\n  end");

    public static final List<String> SPLITTER_CONTRACT_SAMPLES = TOKENIZER_UNIVERSAL_SAMPLES;

    public static final List<String> SPLITTER_PARITY_SAMPLES = TOKENIZER_UNIVERSAL_SAMPLES;

    public static final List<String> QWEN35_SPLITTER_REPRESENTATIVE_SAMPLES =
            List.of(
                    "Hello world",
                    "It's we're THEY'LL I'd",
                    "digits 1234567890",
                    "unicode digits ١٢٣ ۱۲۳ १२३ １２３",
                    "tabs\tand\nnewlines\r\n",
                    "Thai: สวัสดีชาวโลก",
                    "Vietnamese: Xin chào thế giới",
                    "Arabic: مرحبا بالعالم",
                    "NEL only: a\u0085b\u0085c",
                    "line separator: a\u2028b",
                    "paragraph separator: a\u2029b");

    public static final List<String> REGEX_SPLITTER_CORNER_SAMPLES = TOKENIZER_UNIVERSAL_SAMPLES;

    public static final List<String> MODEL_TOKENIZER_AGREEMENT_TEXTS = TOKENIZER_UNIVERSAL_SAMPLES;
}
