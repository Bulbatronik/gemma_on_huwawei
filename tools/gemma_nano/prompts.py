"""Prompt set for distillation: short, everyday questions a watch user might type.

The tiny student cannot learn "everything Gemma knows", so the distillation
domain is deliberately narrow: small talk, simple definitions, facts, tips,
jokes.  ~10-20k prompts; the teacher answers each several times.
"""
import random

SMALL_TALK = [
    "hi", "hello", "hey", "good morning", "good night", "good evening", "how are you", "how are you today",
    "what is your name", "who are you", "what can you do", "are you a robot", "thank you", "thanks",
    "bye", "see you later", "i am bored", "i am tired", "i am happy", "i am sad", "i feel lonely",
    "i am hungry", "i can't sleep", "tell me something nice", "say something funny", "cheer me up",
    "what should i do today", "do you like music", "what is your favorite color", "do you dream",
    "are you smart", "can you help me", "i love you", "you are cool", "what time is it",
    "what day is it", "how old are you", "where do you live", "what do you eat", "do you have friends",
    "motivate me", "give me a compliment", "tell me a secret", "what is the meaning of life",
    "i went for a run", "i just woke up", "it is raining", "it is cold today", "it is hot today",
]

NOUNS = """apple banana orange grape lemon cherry strawberry watermelon tomato potato carrot onion garlic rice bread
cheese milk butter egg honey sugar salt coffee tea water juice soup pizza pasta cake cookie chocolate
cat dog horse cow pig sheep goat chicken duck rabbit mouse lion tiger bear wolf fox deer elephant giraffe
zebra monkey whale dolphin shark fish octopus turtle frog snake bird eagle owl parrot penguin bee ant butterfly
spider tree flower grass leaf forest river lake ocean sea mountain hill desert island beach volcano cloud rain
snow wind storm thunder lightning rainbow sun moon star planet earth mars comet galaxy sky fire ice rock sand
car bus train plane bike boat ship rocket road bridge city village house school library hospital museum park
farm garden kitchen window door chair table bed lamp clock phone computer watch camera radio television book
pen pencil paper map key bag shoe hat shirt coat umbrella ball kite guitar piano drum violin song movie game
doctor teacher nurse farmer pilot chef artist singer friend family baby king queen robot battery magnet
engine wheel coin money bank market shop job holiday birthday party gift letter email internet password
heart brain bone blood muscle skin eye ear nose tooth hand foot sleep dream memory idea question answer
number circle square triangle color music dance art science math history language word story poem joke
summer winter spring autumn morning night week month year time future""".split()

TOPICS = """cats dogs space the ocean the moon the sun stars weather rain food coffee tea pizza fruit vegetables
sleep exercise running walking music books movies games robots computers phones the internet history science
math animals birds fish insects trees flowers mountains rivers winter summer holidays birthdays friendship
family school work money cooking travel trains planes cars bikes the brain the heart water fire electricity
dinosaurs volcanoes earthquakes clouds snow ice chocolate bread cheese honey bees elephants penguins owls
sharks whales lions football tennis chess painting dancing singing languages time clocks maps colors""".split(" ")

ACTIVITIES = [
    "sleep better", "make tea", "make coffee", "boil an egg", "cook rice", "make a sandwich", "learn english",
    "learn to code", "save money", "stay calm", "focus on work", "study for an exam", "make friends",
    "drink more water", "start running", "stretch my back", "relax after work", "wake up early",
    "remember things", "be more productive", "plant a tree", "take care of a cat", "train a dog",
    "write a poem", "tell a good joke", "draw a cat", "keep my phone battery healthy", "stay warm in winter",
    "stay cool in summer", "fall asleep fast", "reduce stress", "be happy", "eat healthy", "lose weight",
    "build muscle", "clean my room", "pack for a trip", "read faster", "learn the guitar", "speak in public",
]

SCIENCE = [
    "why is the sky blue", "why is the sea salty", "why do we sleep", "why do cats purr", "why do birds sing",
    "how do plants grow", "how does the heart work", "how far is the moon", "how hot is the sun",
    "what is gravity", "what is a black hole", "what is dna", "what is an atom", "what is energy",
    "what is light", "what is sound", "what makes thunder", "how do rainbows form", "why do leaves fall",
    "why is grass green", "how do fish breathe", "how do bees make honey", "what is a cloud made of",
    "why does ice float", "what is the biggest animal", "what is the fastest animal", "how many planets are there",
    "what is the largest ocean", "what is the tallest mountain", "how old is the earth", "what is photosynthesis",
    "what causes the seasons", "why do we have day and night", "how does a magnet work", "what is electricity",
    "how do airplanes fly", "how does the internet work", "what is a computer", "what is artificial intelligence",
    "what is a language model", "how do vaccines work", "why do we need water", "why do we dream",
]

TEMPLATES_NOUN = ["what is a {x}", "what is {x}", "tell me about {x}", "describe a {x}", "fun fact about {x}",
                  "is a {x} big", "what color is a {x}", "where can i find a {x}", "why do people like {x}",
                  "write a short poem about a {x}", "what rhymes with {x}"]
TEMPLATES_TOPIC = ["tell me a joke about {x}", "tell me a fact about {x}", "tell me something about {x}",
                   "why are {x} interesting", "give me a fun fact about {x}", "what do you think about {x}",
                   "write one sentence about {x}", "tell me a short story about {x}"]
TEMPLATES_ACT = ["how do i {x}", "how can i {x}", "give me a tip to {x}", "what is the best way to {x}",
                 "help me {x}"]

TEACHER_INSTRUCTION = ("Answer the following in one or two short, simple sentences. "
                       "Use plain words, no lists, no emojis.\n\n")


def all_prompts(seed=0, max_prompts=20000):
    ps = list(SMALL_TALK) + list(SCIENCE)
    ps += [t.format(x=n).replace(" a " + n, (" an " if n[0] in "aeiou" else " a ") + n)
           for n in NOUNS for t in TEMPLATES_NOUN]
    ps += [t.format(x=n.strip()) for n in TOPICS if n.strip() for t in TEMPLATES_TOPIC]
    ps += [t.format(x=a) for a in ACTIVITIES for t in TEMPLATES_ACT]
    rnd = random.Random(seed)
    rnd.shuffle(ps)
    ps = ps[:max_prompts]
    # light surface variation (the watch keyboard produces lowercase text)
    out = []
    for p in ps:
        r = rnd.random()
        if r < 0.4:
            p = p[0].upper() + p[1:] + ("?" if p.split()[0] in ("what", "why", "how", "is", "are", "do", "where", "who", "can") else "")
        elif r < 0.6:
            p = p + "?"
        out.append(p)
    return out
