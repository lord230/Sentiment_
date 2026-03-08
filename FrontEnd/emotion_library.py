# Created by LORD
#
# emotion_library.py
#
# Hand-written seed sentences for every emotion.
# Each entry:  (text, emotion, sentiment_polarity, intensity)
#
#   sentiment_polarity : "positive" | "negative" | "neutral"
#   intensity          : "low" | "medium" | "high"
#
# BALANCE TARGETS:
#   positive : ~45%   negative : ~45%   neutral : ~10%
#
# Rule: every emotion that can appear in a sarcastic-positive
# context MUST have positive-polarity seeds so the positive
# sub-index can find them.

EMOTION_SEEDS = [

    # ══════════════════════════════════════════════════════
    # POSITIVE POLARITY
    # ══════════════════════════════════════════════════════

    # ── JOY ─────────────────────────────────────────────────────────────
    ("I am so happy right now, everything feels perfect",               "Joy",              "positive", "high"),
    ("This made my whole day, I absolutely love it",                    "Joy",              "positive", "high"),
    ("I feel really good about how everything turned out",              "Joy",              "positive", "medium"),
    ("That was a pleasant surprise, I genuinely enjoyed it",            "Joy",              "positive", "medium"),
    ("Things are going alright, I feel pretty okay",                    "Joy",              "positive", "low"),
    ("Not bad at all, actually quite nice",                             "Joy",              "positive", "low"),

    # ── EXCITEMENT ──────────────────────────────────────────────────────
    ("I cannot wait, this is going to be absolutely incredible",        "Excitement",       "positive", "high"),
    ("Oh my god YES, I have been waiting for this forever",             "Excitement",       "positive", "high"),
    ("I am so pumped up and ready, let us go",                          "Excitement",       "positive", "high"),
    ("Really looking forward to this, it sounds amazing",               "Excitement",       "positive", "medium"),
    ("That sounds fun, I am genuinely interested",                      "Excitement",       "positive", "low"),

    # ── GRATITUDE ───────────────────────────────────────────────────────
    ("Thank you so much, this truly means the world to me",             "Gratitude",        "positive", "high"),
    ("I am deeply grateful for everything you have done for me",        "Gratitude",        "positive", "high"),
    ("I really appreciate your help, thank you sincerely",              "Gratitude",        "positive", "medium"),
    ("Thanks, that was genuinely kind of you",                          "Gratitude",        "positive", "low"),

    # ── LOVE / AFFECTION ────────────────────────────────────────────────
    ("I love you more than anything in this world",                     "Love",             "positive", "high"),
    ("You are the most important person in my life",                    "Love",             "positive", "high"),
    ("I care about you so deeply and I always will",                    "Love",             "positive", "high"),
    ("I really cherish every moment we spend together",                 "Love",             "positive", "medium"),
    ("You are such a good friend and I appreciate you so much",         "Love",             "positive", "low"),

    # ── PRIDE ───────────────────────────────────────────────────────────
    ("I am so proud of what I have achieved, I worked so hard for this","Pride",            "positive", "high"),
    ("We did it, I always knew we could pull this off together",        "Pride",            "positive", "high"),
    ("That was a solid performance and I feel great about it",          "Pride",            "positive", "medium"),
    ("I think I handled that pretty well actually",                     "Pride",            "positive", "low"),

    # ── HOPE ────────────────────────────────────────────────────────────
    ("I truly believe things are going to get so much better from here","Hope",             "positive", "high"),
    ("There is still a real chance and I am holding on to it",          "Hope",             "positive", "medium"),
    ("Maybe it will all work out, I am staying optimistic",             "Hope",             "positive", "low"),

    # ── AMUSEMENT ───────────────────────────────────────────────────────
    ("That is absolutely hilarious, I literally cannot stop laughing",  "Amusement",        "positive", "high"),
    ("Haha that was genuinely funny, I really needed that laugh",       "Amusement",        "positive", "medium"),
    ("That gave me a little chuckle, pretty amusing",                   "Amusement",        "positive", "low"),

    # ── ADMIRATION / AWE ────────────────────────────────────────────────
    # NOTE: Expanded with compliment-style seeds so positive queries
    #       (incl. sarcastic-positive about appearance/ability) land here.
    ("That is breathtaking, I am completely in awe",                    "Admiration",       "positive", "high"),
    ("I have so much respect for what they accomplished",               "Admiration",       "positive", "high"),
    ("She is already so perfect, what is there even left to see",       "Admiration",       "positive", "high"),
    ("What is there to look at, she is already absolutely flawless",    "Admiration",       "positive", "high"),
    ("There is nothing left to prove, she is simply perfect",           "Admiration",       "positive", "high"),
    ("He is already incredible, there is nothing more to add",          "Admiration",       "positive", "high"),
    ("They are so talented it almost does not seem real",               "Admiration",       "positive", "high"),
    ("I cannot find a single flaw, everything about them is perfect",   "Admiration",       "positive", "high"),
    ("That was genuinely impressive, I am truly in awe",                "Admiration",       "positive", "medium"),
    ("Not bad at all, I really appreciate the effort",                  "Admiration",       "positive", "low"),

    # ── RELIEF ──────────────────────────────────────────────────────────
    ("Thank god that is finally over, I can breathe again",             "Relief",           "positive", "high"),
    ("I was so worried but it all worked out perfectly in the end",     "Relief",           "positive", "high"),
    ("So glad that is sorted, I was really stressing about it",         "Relief",           "positive", "medium"),
    ("Phew, okay that was not as bad as I thought it would be",         "Relief",           "positive", "low"),

    # ── NOSTALGIA (positive) ─────────────────────────────────────────────
    ("I miss the old days so much, everything felt so much simpler",    "Nostalgia",        "positive", "high"),
    ("Thinking about those times makes me smile and ache at once",      "Nostalgia",        "positive", "medium"),
    ("It reminds me of better and happier times gone by",               "Nostalgia",        "positive", "low"),

    # ── EMPATHY / COMPASSION ─────────────────────────────────────────────
    ("My heart truly breaks for what they are going through right now", "Empathy",          "positive", "high"),
    ("I can feel how much pain they are in and I want to help them",    "Empathy",          "positive", "high"),
    ("I really feel for them, that sounds incredibly difficult",        "Empathy",          "positive", "medium"),
    ("That must be really hard, I sincerely hope they are okay",        "Empathy",          "positive", "low"),

    # ── SURPRISE (positive) ─────────────────────────────────────────────
    ("I did not see that coming at all, what a wonderful surprise",     "Surprise",         "positive", "high"),
    ("That was totally unexpected but in the most wonderful way",       "Surprise",         "positive", "medium"),
    ("Oh wow I was not expecting that at all, pleasantly surprised",    "Surprise",         "positive", "low"),

    # ── CALM / CONTENTMENT (positive) ───────────────────────────────────
    ("I feel completely at peace with everything right now",            "Calm",             "positive", "medium"),
    ("Life is good, quiet and simple, and I am genuinely content",      "Calm",             "positive", "low"),

    # ══════════════════════════════════════════════════════
    # NEGATIVE POLARITY
    # ══════════════════════════════════════════════════════

    # ── SADNESS ─────────────────────────────────────────────────────────
    ("I am absolutely heartbroken and I do not know what to do",        "Sadness",          "negative", "high"),
    ("I cried for hours, I feel completely empty inside",               "Sadness",          "negative", "high"),
    ("This is really upsetting, I feel so completely down",             "Sadness",          "negative", "high"),
    ("I have been feeling really low and sad lately",                   "Sadness",          "negative", "medium"),
    ("Things have not been great, I have been feeling a bit sad",       "Sadness",          "negative", "medium"),
    ("I am a little down today, I am not sure exactly why",             "Sadness",          "negative", "low"),

    # ── GRIEF ───────────────────────────────────────────────────────────
    ("I lost someone I truly love and the pain is completely unbearable","Grief",           "negative", "high"),
    ("The grief is overwhelming, I simply cannot function at all",      "Grief",            "negative", "high"),
    ("I miss them every single day and it never ever gets easier",      "Grief",            "negative", "high"),
    ("I am still trying hard to cope with losing them",                 "Grief",            "negative", "medium"),

    # ── ANGER ───────────────────────────────────────────────────────────
    ("I am absolutely furious, this is completely unacceptable",        "Anger",            "negative", "high"),
    ("How dare they do this to me, I am absolutely livid right now",    "Anger",            "negative", "high"),
    ("This makes me so angry I genuinely cannot think straight",        "Anger",            "negative", "high"),
    ("I am pretty annoyed about what just happened",                    "Anger",            "negative", "medium"),
    ("That was frustrating and completely unnecessary",                 "Anger",            "negative", "medium"),
    ("That was slightly irritating honestly",                           "Anger",            "negative", "low"),

    # ── FRUSTRATION ─────────────────────────────────────────────────────
    ("Nothing is working and I am at my absolute breaking point",       "Frustration",      "negative", "high"),
    ("I keep trying and it keeps failing, I am ready to give up",       "Frustration",      "negative", "high"),
    ("This is so frustrating, I simply cannot get it right",            "Frustration",      "negative", "medium"),
    ("It is taking forever and I am really losing all patience",        "Frustration",      "negative", "medium"),
    ("A bit annoying but I will somehow manage it",                     "Frustration",      "negative", "low"),

    # ── FEAR / ANXIETY ──────────────────────────────────────────────────
    ("I am absolutely terrified, I do not feel safe at all",            "Fear",             "negative", "high"),
    ("The anxiety is crushing me and I genuinely cannot breathe",       "Fear",             "negative", "high"),
    ("I am really scared about what might happen next",                 "Fear",             "negative", "high"),
    ("Feeling pretty anxious and very much on edge today",              "Fear",             "negative", "medium"),
    ("A bit nervous about how all of this will go",                     "Fear",             "negative", "low"),

    # ── DISGUST ─────────────────────────────────────────────────────────
    ("That is absolutely revolting, I am completely disgusted",         "Disgust",          "negative", "high"),
    ("I find this morally reprehensible and deeply appalling",          "Disgust",          "negative", "high"),
    ("That was gross and made me deeply uncomfortable",                 "Disgust",          "negative", "medium"),
    ("Honestly that put me off quite a bit",                            "Disgust",          "negative", "low"),

    # ── CONTEMPT ────────────────────────────────────────────────────────
    ("I have absolutely zero respect for that person",                  "Contempt",         "negative", "high"),
    ("They are beneath my notice, completely pathetic",                 "Contempt",         "negative", "high"),
    ("I honestly do not think much of them at all",                     "Contempt",         "negative", "medium"),

    # ── DISAPPOINTMENT ──────────────────────────────────────────────────
    ("I had such high hopes and this let me down completely",           "Disappointment",   "negative", "high"),
    ("I expected so much better from them honestly",                    "Disappointment",   "negative", "high"),
    ("That was really disappointing, I thought it would be better",     "Disappointment",   "negative", "medium"),
    ("Not what I had hoped for but okay I suppose",                     "Disappointment",   "negative", "low"),

    # ── LONELINESS ──────────────────────────────────────────────────────
    ("I feel completely alone and nobody truly understands me",         "Loneliness",       "negative", "high"),
    ("Nobody reaches out anymore, I feel completely invisible",         "Loneliness",       "negative", "high"),
    ("Feeling isolated and totally disconnected from everyone",         "Loneliness",       "negative", "medium"),
    ("A bit lonely lately, really missing people I care about",         "Loneliness",       "negative", "low"),

    # ── SHAME / EMBARRASSMENT ───────────────────────────────────────────
    ("I am so deeply ashamed of what I did, I cannot face anyone",      "Shame",            "negative", "high"),
    ("That was mortifying, I genuinely wanted to disappear",            "Shame",            "negative", "high"),
    ("I feel really bad about how I handled the whole thing",           "Shame",            "negative", "medium"),
    ("That was honestly a bit embarrassing",                            "Shame",            "negative", "low"),

    # ── GUILT ───────────────────────────────────────────────────────────
    ("I feel terrible about what I did, I should have known better",    "Guilt",            "negative", "high"),
    ("The guilt is eating me alive and I cannot forgive myself",        "Guilt",            "negative", "high"),
    ("I feel bad about it and I wish I had acted differently",          "Guilt",            "negative", "medium"),
    ("I probably really should not have done that",                     "Guilt",            "negative", "low"),

    # ── ENVY / JEALOUSY ─────────────────────────────────────────────────
    ("I am so jealous, why does everything good happen to them",        "Envy",             "negative", "high"),
    ("I wish I had what they have, it is simply not fair",              "Envy",             "negative", "medium"),
    ("Slightly envious but trying genuinely not to dwell on it",        "Envy",             "negative", "low"),

    # ── SHOCK ───────────────────────────────────────────────────────────
    ("I cannot believe that just happened, I am completely shocked",    "Shock",            "negative", "high"),
    ("That came from absolutely nowhere, I am totally blindsided",      "Shock",            "negative", "high"),
    ("I was not at all prepared for that",                              "Shock",            "negative", "medium"),

    # ── OVERWHELM ───────────────────────────────────────────────────────
    ("There is too much happening at once and I genuinely cannot cope", "Overwhelm",        "negative", "high"),
    ("I am drowning in responsibilities and I do not know where to start","Overwhelm",      "negative", "high"),
    ("Feeling really overwhelmed with everything on my plate",          "Overwhelm",        "negative", "medium"),

    # ── SARCASM / IRONY ─────────────────────────────────────────────────
    ("Oh wow what a totally shocking and unexpected surprise",          "Sarcasm",          "negative", "high"),
    ("Yeah sure because that always works out so perfectly",            "Sarcasm",          "negative", "high"),
    ("Oh great another brilliant idea from the genius in charge",       "Sarcasm",          "negative", "high"),
    ("Sure, because I obviously have nothing better to do than this",   "Sarcasm",          "negative", "medium"),
    ("Oh how wonderful, exactly what I always dreamed of",              "Sarcasm",          "negative", "medium"),
    ("Clearly you are not ready, as always",                            "Sarcasm",          "negative", "medium"),
    ("Oh absolutely, because everything you do is just perfect",        "Sarcasm",          "negative", "low"),

    # ── CYNICISM ────────────────────────────────────────────────────────
    ("Nothing ever really changes, everything is broken and always will","Cynicism",        "negative", "high"),
    ("They keep promising things will be different but they never are", "Cynicism",         "negative", "high"),
    ("I stopped expecting good outcomes a very long time ago",          "Cynicism",         "negative", "medium"),

    # ── BOREDOM (negative) ──────────────────────────────────────────────
    ("I am so bored I want to scream, absolutely nothing is happening", "Boredom",          "negative", "high"),
    ("This is painfully dull and I simply cannot focus",                "Boredom",          "negative", "medium"),

    # ══════════════════════════════════════════════════════
    # NEUTRAL POLARITY
    # ══════════════════════════════════════════════════════

    # ── CALM (neutral) ──────────────────────────────────────────────────
    ("Nothing special is happening, just a completely normal day",      "Calm",             "neutral",  "low"),

    # ── BOREDOM (neutral) ───────────────────────────────────────────────
    ("A bit bored today, just looking for something to do",             "Boredom",          "neutral",  "low"),

    # ── CONFUSION ───────────────────────────────────────────────────────
    ("I have absolutely no idea what is going on here",                 "Confusion",        "neutral",  "high"),
    ("This makes no sense to me whatsoever",                            "Confusion",        "neutral",  "medium"),
    ("I am a bit lost but genuinely trying to figure it out",           "Confusion",        "neutral",  "low"),

    # ── INDIFFERENCE ────────────────────────────────────────────────────
    ("I genuinely could not care less about any of this",               "Indifference",     "neutral",  "high"),
    ("Not really bothered either way",                                  "Indifference",     "neutral",  "medium"),
    ("I have no strong feelings about it at all",                       "Indifference",     "neutral",  "low"),

    # ── NOSTALGIA (neutral) ──────────────────────────────────────────────
    ("It kind of reminds me of times long gone by",                     "Nostalgia",        "neutral",  "low"),
]