//! What the word list doesn't say: which words are common (a fix to one is
//! likelier), everyday words newer than it, and contractions typed without
//! their apostrophe.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Common English words, most common first.
pub const COMMON: &[&str] = &[
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as",
    "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
    "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into",
    "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only",
    "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "are", "was",
    "were", "been", "has", "had", "did", "does", "said", "made", "here", "where", "why", "should", "very",
    "through", "before", "much", "too", "more", "many", "those", "own", "same", "still", "while", "need",
    "find", "tell", "ask", "seem", "feel", "try", "leave", "call", "keep", "let", "begin", "help", "show",
    "hear", "play", "run", "move", "live", "believe", "bring", "happen", "write", "provide", "sit", "stand",
    "lose", "pay", "meet", "include", "continue", "set", "learn", "change", "lead", "understand", "watch",
    "follow", "stop", "create", "speak", "read", "allow", "add", "spend", "grow", "open", "walk", "win",
    "offer", "remember", "love", "consider", "appear", "buy", "wait", "serve", "die", "send", "expect",
    "build", "stay", "fall", "cut", "reach", "kill", "remain", "thing", "things", "man", "world", "life",
    "hand", "part", "child", "eye", "woman", "place", "week", "case", "point", "government", "company",
    "number", "group", "problem", "fact", "right", "left", "old", "great", "big", "high", "different", "small",
    "large", "next", "early", "young", "important", "few", "public", "bad", "able", "last", "long", "little",
    "never", "always", "again", "really", "something", "nothing", "everything", "anything", "someone",
    "everyone", "maybe", "already", "today", "tomorrow", "yesterday", "yes", "please", "thanks", "thank",
    "sorry", "okay", "hello", "hi", "hey", "file", "files", "code", "test", "tests", "fix", "bug", "error",
    "errors", "line", "lines", "function", "build", "run", "check", "change", "changes", "update", "add",
    "remove", "delete", "make", "sure", "instead", "without", "within", "between", "under", "around", "again",
    "against", "during", "until", "since", "though", "although", "however", "whether", "either", "neither",
    "each", "every", "both", "another", "such", "own", "enough", "quite", "rather", "almost", "probably",
    "actually", "exactly", "especially", "usually", "really", "definitely", "separate", "receive", "believe",
    "necessary", "different", "possible", "available", "information", "example", "question", "answer",
    "problem", "reason", "result", "system", "program", "computer", "keyboard", "screen", "window", "button",
    "message", "name", "word", "words", "letter", "text", "page", "hello", "world", "would", "could", "should",
    "might", "must", "shall", "there", "they're", "you're", "it's", "can't", "don't", "won't", "isn't", "aren't",
    "doesn't", "didn't", "wasn't", "weren't", "couldn't", "shouldn't", "wouldn't", "haven't", "hasn't",
    "hadn't", "i'm", "i've", "i'll", "i'd", "we're", "we've", "they've", "that's", "what's", "let's", "here's",
    "there's", "where's", "who's", "how's", "weird", "friend", "because", "said", "which", "their", "receive",
    "really", "know", "about", "think", "write", "wrong", "right", "true", "false", "start", "end", "stop",
    "first", "second", "third", "next", "previous", "same", "other", "new", "old", "now", "then", "soon",
    "later", "often", "sometimes", "never", "ever", "once", "twice", "here", "there", "everywhere", "nowhere",
    "somewhere", "inside", "outside", "above", "below", "near", "far", "away", "back", "forward", "down",
    "together", "alone", "along", "across", "behind", "beside", "beyond", "into", "onto", "upon", "toward",
    "towards", "among", "per", "via", "etc", "and", "or", "but", "nor", "yet", "so", "for", "because",
    "unless", "though", "accommodate", "occurred", "occurrence", "until", "tomorrow", "beginning", "argument",
];

/// Everyday words newer than the word list (or its inflections of them it
/// can't work out).
pub const NEWER: &[&str] = &[
    "email", "emails", "emailed", "internet", "website", "websites", "online", "offline", "app", "apps",
    "smartphone", "smartphones", "laptop", "laptops", "wifi", "login", "logins", "logout", "username",
    "usernames", "password", "passwords", "download", "downloads", "downloaded", "upload", "uploads",
    "uploaded", "software", "hardware", "browser", "browsers", "server", "servers", "database", "databases",
    "online", "blog", "blogs", "video", "videos", "podcast", "podcasts", "selfie", "hashtag", "emoji",
    "emojis", "google", "googled", "iphone", "ipad", "mac", "macos", "linux", "unix", "windows", "github",
    "gitlab", "repo", "repos", "commit", "commits", "committed", "branch", "branches", "merge", "merged",
    "rebase", "deploy", "deployed", "deployment", "debug", "debugged", "debugging", "config", "configs",
    "terminal", "terminals", "shell", "script", "scripts", "api", "apis", "json", "yaml", "toml", "html",
    "css", "url", "urls", "http", "https", "ssh", "sudo", "localhost", "async", "await", "backend", "frontend",
    "framework", "frameworks", "refactor", "refactored", "refactoring", "runtime", "timestamp", "timestamps",
    "workflow", "workflows", "dataset", "datasets", "metadata", "plugin", "plugins", "screenshot",
    "screenshots", "smartwatch", "touchscreen", "startup", "startups", "covid", "ok", "okay", "lol", "btw",
    "fyi", "gonna", "wanna", "gotta", "yeah", "nope", "yep", "hmm", "cool", "awesome", "stuff", "guy", "guys",
    "kid", "kids", "phone", "phones", "text", "texted", "texting", "click", "clicked", "clicking", "tap",
    "tapped", "tapping", "scroll", "scrolled", "scrolling", "swipe", "swiped", "keyboard", "keyboards",
    "dialog", "dialogs", "codebase", "todo", "todos", "readme", "changelog", "env", "dev", "devs", "prod", "staging", "docker",
    "kubernetes", "npm", "cargo", "rust", "python", "javascript", "typescript", "golang", "java", "swift",
    "kotlin", "claude", "anthropic", "openai", "chatgpt", "ai", "llm", "llms", "gpu", "gpus", "cpu", "cpus",
];

/// Irregular forms the word list lacks (it lists "hold", not "held").
pub const IRREGULAR: &[&str] = &[
    "arose", "arisen", "awoke", "awoken", "bore", "borne", "beat", "beaten", "became", "began", "begun", "bent",
    "bet", "bid", "bit", "bitten", "bled", "blew", "blown", "broke", "broken", "bred", "brought", "built",
    "burnt", "burst", "bought", "caught", "chose", "chosen", "clung", "came", "cost", "crept", "dealt", "dug",
    "did", "done", "drew", "drawn", "dreamt", "drank", "drunk", "drove", "driven", "ate", "eaten", "fell",
    "fallen", "fed", "felt", "fought", "found", "fled", "flung", "flew", "flown", "forbade", "forbidden",
    "forgot", "forgotten", "forgave", "forgiven", "froze", "frozen", "got", "gotten", "gave", "given", "went",
    "gone", "ground", "grew", "grown", "hung", "had", "heard", "hid", "hidden", "hit", "held", "hurt", "kept",
    "knelt", "knew", "known", "laid", "led", "leapt", "learnt", "left", "lent", "lay", "lain", "lit", "lost",
    "made", "meant", "met", "paid", "proved", "proven", "put", "quit", "read", "rid", "rode", "ridden", "rang",
    "rung", "rose", "risen", "ran", "said", "saw", "seen", "sought", "sold", "sent", "set", "sewn", "shook",
    "shaken", "shed", "shone", "shot", "shown", "shrank", "shrunk", "shut", "sang", "sung", "sank", "sunk",
    "sat", "slept", "slid", "slung", "spoke", "spoken", "sped", "spent", "spilt", "spun", "spat", "split",
    "spread", "sprang", "sprung", "stood", "stole", "stolen", "stuck", "stung", "stank", "stunk", "struck",
    "strung", "strove", "swore", "sworn", "swept", "swam", "swum", "swung", "took", "taken", "taught", "tore",
    "torn", "told", "thought", "threw", "thrown", "understood", "woke", "woken", "wore", "worn", "wove",
    "woven", "wept", "won", "wound", "wrung", "wrote", "written", "children", "men", "women", "people", "feet",
    "teeth", "mice", "geese", "data", "criteria", "phenomena", "indices", "matrices", "vertices", "analyses",
    "better", "best", "worse", "worst", "further", "furthest", "less", "least", "more", "most", "nerd", "nerds",
];

/// British spellings the (American) word list lacks.
pub const BRITISH: &[&str] = &[
    "colour", "colours", "coloured", "favour", "favours", "favourite", "favourites", "honour", "labour",
    "neighbour", "neighbours", "behaviour", "behaviours", "flavour", "humour", "rumour", "harbour", "licence",
    "licences", "defence", "offence", "centre", "centres", "metre", "metres", "litre", "theatre", "fibre",
    "organise", "organised", "organising", "organisation", "realise", "realised", "recognise", "recognised",
    "apologise", "analyse", "analysed", "catalogue", "dialogue", "programme", "programmes", "grey", "cheque",
    "travelled", "travelling", "cancelled", "cancelling", "modelled", "modelling", "labelled", "labelling",
    "jewellery", "aluminium", "mum", "maths", "whilst", "amongst", "learnt", "spelt", "optimise", "optimised",
    "initialise", "initialised", "customise", "customised", "synchronise", "summarise", "minimise", "maximise",
];

/// The rank of a common word (0: the most common).
pub fn common_rank(w: &str) -> Option<usize> {
    static RANKS: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    let ranks = RANKS.get_or_init(|| {
        let mut m = HashMap::new();
        for (i, w) in COMMON.iter().enumerate() {
            m.entry(*w).or_insert(i);
        }
        m
    });
    ranks.get(w).copied()
}

/// A contraction typed without its apostrophe, or "i" for "I". Only those
/// that aren't words themselves ("cant", "wont", "well", "were", "hell"
/// are).
pub fn contraction(w: &str) -> Option<&'static str> {
    Some(match w {
        "i" => "I",
        "im" => "I'm",
        "ive" => "I've",
        "id" => return None,
        "dont" => "don't",
        "doesnt" => "doesn't",
        "didnt" => "didn't",
        "isnt" => "isn't",
        "arent" => "aren't",
        "wasnt" => "wasn't",
        "werent" => "weren't",
        "havent" => "haven't",
        "hasnt" => "hasn't",
        "hadnt" => "hadn't",
        "couldnt" => "couldn't",
        "shouldnt" => "shouldn't",
        "wouldnt" => "wouldn't",
        "mustnt" => "mustn't",
        "youre" => "you're",
        "youve" => "you've",
        "youll" => "you'll",
        "theyre" => "they're",
        "theyve" => "they've",
        "theyll" => "they'll",
        "weve" => "we've",
        "thats" => "that's",
        "whats" => "what's",
        "wheres" => "where's",
        "theres" => "there's",
        "heres" => "here's",
        "whos" => "who's",
        "hows" => "how's",
        "itll" => "it'll",
        "lets" => return None,
        _ => return None,
    })
}
