from sarcasmscope.text.emoji import emoji_features, extract_emojis
from sarcasmscope.text.langid import guess_language
from sarcasmscope.text.normalize import dedup_key, normalize
from sarcasmscope.text.scripts import code_mix_index, dominant_script, tokenize_words

WORDS = {
    "Devanagari": "नमस्ते", "Telugu": "ధన్యవాదాలు", "Tamil": "நன்றி", "Bengali": "ধন্যবাদ",
    "Gujarati": "ધન્યવાદ", "Kannada": "ಧನ್ಯವಾದ", "Malayalam": "നന്ദി", "Gurmukhi": "ਧੰਨਵਾਦ",
    "Arabic": "شکریہ", "Odia": "ଧନ୍ୟବାଦ", "Latin": "thanks",
}


def test_tc01_script_detection_all_scripts():
    for script, word in WORDS.items():
        assert dominant_script(word) == script, (script, word)


def test_tc02_tokenizer_keeps_indic_words_whole():
    assert tokenize_words("वाह, क्या बात है!") == ["वाह", "क्या", "बात", "है"]
    assert tokenize_words("great😂 !!!") == ["great"]


def test_tc03_code_mix_index():
    mixed = code_mix_index("wah क्या बात है")
    assert len(mixed["token_scripts"]) == 2 and mixed["cmi"] > 0
    assert code_mix_index("क्या बात है")["cmi"] == 0


def test_tc04_normalize_removes_label_hashtags_and_masks():
    out = normalize("Soooo great #sarcasm #life @bob https://x.co/a !!!")
    assert "sarcasm" not in out and "life" in out
    assert "<user>" in out and "<url>" in out and "!!!" in out
    assert "soo" in out and "sooo" not in out


def test_tc05_normalize_keeps_emoji_and_zwj():
    assert "🙄" in normalize("wah 🙄")
    assert "🙄" not in normalize("wah 🙄", keep_emoji=False)
    s = "क\u200dष"
    assert "\u200d" in normalize(s)


def test_tc06_emoji_extraction():
    assert extract_emojis("a 👨\u200d👩\u200d👧 b 🙄") == ["👨\u200d👩\u200d👧", "🙄"]
    assert extract_emojis("क\u200dष") == []
    assert emoji_features("wah 🙄😂")["n_irony_emoji"] == 1


def test_tc07_language_guess():
    assert guess_language("ధన్యవాదాలు")["language"] == "te"
    assert guess_language("आज मौसम बहुत अच्छा है और मैं खुश हूँ")["language"] == "hi"
    assert guess_language("आज हवामान छान आहे आणि मी खूप खुश आहे")["language"] == "mr"
    g = guess_language("wah kya baat hai yaar")
    assert g["romanized"] and g["language"] == "hi"


def test_tc08_dedup_key_ignores_case_emoji_punct():
    assert dedup_key("Wah!! Kya baat hai 🙄") == dedup_key("wah kya baat hai")
