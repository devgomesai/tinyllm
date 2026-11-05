"""Deterministic dataset expansion for TinyGPT 2.0.

Generates high-quality, diverse question/answer examples across all required
categories. All content is authored here (or computed arithmetically) and is
reproducible with fixed seeds. Characters used are kept inside the existing
86-character vocabulary of data.txt (no backslash, no '#', no '%', no unicode)
so that both the char-level and BPE models, and the v1 baseline, can encode
the questions without surprises.

Question paraphrasis reuses the wrapper style of the original corpus to teach
rewording robustness; surface questions are checked for duplicates against
data/data.txt during the pipeline's dedupe step.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from trainer.dataset import (
    NUM_RE,
    PROJECT_DIR,
    Example,
    base_question,
    normalize_question,
)

DATA_FILE = PROJECT_DIR / "data" / "data.txt"

WHAT_WRAPPERS = [
    "What is {b}?",
    "Can I ask, what is {b}?",
    "Please tell me, what is {b}?",
    "I was wondering, what is {b}?",
    "What do you know about {b}?",
    "What's {b}?",
    "Can you explain what {b} is?",
    "Define {baki}",
    "Explain {baki}",
    "Tell me about {baki}",
]
HOW_WRAPPERS = [
    "How do I {b}?",
    "Can I ask, how do I {b}?",
    "Please tell me, how do I {b}?",
    "I was wondering, how do I {b}?",
    "How does one {b}?",
]
WHY_WRAPPERS = [
    "Why do {b}?",
    "Can you explain why {b}?",
    "Please tell me, why {b}?",
]


def _capitalize(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def make_what(category: str, base: str, answer: str) -> list[Example]:
    """Generate up to three paraphrased question variants for one fact."""
    b = base.strip()
    baki = b if b.startswith(("a ", "an ", "the ")) else b
    out = []
    seen = set()
    for tmpl in (WHAT_WRAPPERS[0], WHAT_WRAPPERS[1], WHAT_WRAPPERS[5]):
        q = tmpl.format(b=b, baki=baki)
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(Example(q, answer, source="expansion/facts",
                               category=category))
    return out


def make_how(category: str, base: str, answer: str) -> list[Example]:
    b = base.strip()
    out = []
    for tmpl in HOW_WRAPPERS[:3]:
        q = tmpl.format(b=b)
        out.append(Example(q, answer, source="expansion/facts", category=category))
    return out


# ---------------------------------------------------------------------------
# Fact bank. (category, type, base, answer)
# ---------------------------------------------------------------------------

FACTS = [
    # ---------------- Python fundamentals ----------------
    ("python", "what", "a tuple", "A tuple is an ordered collection, like (1, 2, 3), that cannot be changed after it is made."),
    ("python", "what", "a dictionary", "A dictionary stores pairs of keys and values, like {\"name\": 4} maps a name to a number."),
    ("python", "what", "a set", "A set is a collection of unique items with no order, like {1, 2, 3}."),
    ("python", "what", "a slice", "A slice picks a part of a sequence, like names[0:2] gives the first two items."),
    ("python", "what", "an f-string", "An f-string builds text with values inside, like f(name) inserts the value of the name variable."),
    ("python", "what", "the range function", "The range function makes a sequence of numbers, like range(3) gives 0, 1, 2."),
    ("python", "what", "the len function", "The len function returns how many items a sequence holds, so len(abc) is 3."),
    ("python", "what", "the type function", "The type function tells you the kind of a value, such as int, str, or list."),
    ("python", "what", "a lambda function", "A lambda function is a small unnamed function written in one line, like lambda x: x + 1."),
    ("python", "what", "None", "None is a special value that means no value or nothing was found."),
    ("python", "what", "a boolean", "A boolean is a value that is either True or False."),
    ("python", "what", "an import statement", "An import statement loads code from another file or library so you can use it."),
    ("python", "what", "the str function", "The str function turns a value into text, so 5 becomes the characters 5."),
    ("python", "what", "the float function", "The float function turns a value into a decimal number, like 2.5."),
    ("python", "what", "a module", "A module is a file with Python code that you can import and reuse."),
    ("python", "what", "a library", "A library is a collection of modules made by others that you can install and import."),
    ("python", "what", "the zip function", "The zip function pairs items of two sequences, like [1, 2] and [3, 4] become (1, 3) and (2, 4)."),
    ("python", "what", "the enumerate function", "The enumerate function gives each item of a list together with its position number."),
    ("python", "what", "string chaining", "String chaining joins texts with the plus sign, like \"cat\" + \"dog\" gives catdog."),
    ("python", "what", "a default argument", "A default argument is a value a function uses when the caller gives no other value."),
    ("python", "what", "a keyword argument", "A keyword argument passes a value by name, like greeting(name equal Anna)."),
    ("python", "what", "operator precedence", "Operator precedence decides which math operator runs first, so times runs before plus."),
    ("python", "what", "the in operator", "The in operator checks whether an item exists inside a sequence and returns True or False."),
    ("python", "what", "a truthy value", "A truthy value is any value that counts as True in a condition, like a nonzero number or a nonempty string."),
    ("python", "what", "a dict key error", "A dict key error happens when you look up a key that the dictionary does not contain."),
    ("python", "how", "sort a list in Python", "Use the sort method, like numbers.sort(), or the sorted function to get a new sorted list."),
    ("python", "how", "swap two variables in Python", "Write a, b = b, a on one line, which swaps both variables at once."),
    ("python", "how", "convert text to a number in Python", "Use the int function, like int(str_text), after the text holds only digits."),
    ("python", "how", "make an empty list in Python", "Write empty_box = [] with square brackets and no items inside."),
    ("python", "how", "check the type of a value in Python", "Use the type function, like print(type(x)), and Python shows the class of the value."),
    ("python", "how", "join a list of words in Python", "Use the join method with a space, like quote words dot join words, to make one text."),
    ("python", "how", "read input from the user in Python", "Use the input function, and remember it always gives back text, never a number."),
    ("python", "how", "write a comment in Python", "Start the line with the hash symbol, and Python ignores everything after it."),
    ("python", "how", "make a random choice in Python", "Import the random library and call random.choice(my_list) to pick one item."),
    ("python", "how", "repeat code ten times in Python", "Write a for loop over range(10) and put the code inside, indented by four spaces."),
    ("python", "how", "compare two strings in Python", "Use the equals sign twice, like text_a == text_b, which is True when both texts match exactly."),
    ("python", "why", "statements in Python need correct indentation", "Python uses indentation instead of braces, so wrong spacing makes the code invalid or changes its meaning."),
    ("python", "why", "mutable default arguments are risky in Python", "The default list is created once and shared between calls, so old values stay inside it."),
    ("python", "why", "numbers do not need quotes in Python", "Quotes make text, and text cannot be used in math, so numbers stay unquoted for arithmetic."),

    # ---------------- Python exercises / program structure ----------------
    ("python_exercise", "what", "a function header", "A function header is the first line of a function with def, the name, and the parameters."),
    ("python_exercise", "what", "a return value", "A return value is the result a function hands back to the code that called it."),
    ("python_exercise", "what", "a counter variable", "A counter variable is a number that increases each time a loop runs to count the work done."),
    ("python_exercise", "what", "an accumulator", "An accumulator is a variable that collects a running total while a loop runs."),
    ("python_exercise", "what", "a sentinel value", "A sentinel value is a special value, like an empty string, that tells a loop to stop."),
    ("python_exercise", "what", "an even number check", "An even number check uses the remainder operator with 2, so a number with remainder 0 is even."),
    ("python_exercise", "how", "count words in a sentence with Python", "Use the split method to break the text into words and count the items of the result."),
    ("python_exercise", "how", "print numbers and text together in Python", "Use an f-string with the f marker and place the variable inside curly braces."),
    ("python_exercise", "how", "sum the numbers in a list with Python", "Call the sum function with the list, like sum(my_list), and it returns the total."),
    ("python_exercise", "how", "find the smallest number in a list with Python", "Use the min function, like min(numbers), which returns the smallest value."),
    ("python_exercise", "how", "check if a number is inside a list with Python", "Use the in operator, like 3 in numbers, which is True when the item exists."),
    ("python_exercise", "how", "reverse a string with Python", "Use slicing with a negative step, like text[::-1], which reads the text backwards."),
    ("python_exercise", "how", "store words and their counts with Python", "Use a dictionary and add one to the value for each word you meet while looping."),
    ("python_exercise", "what", "a trace table", "A trace table records each variable after every step of a program, which helps you follow loops."),
    ("python_exercise", "what", "pseudocode", "Pseudocode is a plain-language sketch of a program using normal sentences instead of real code."),

    # ---------------- PyTorch ----------------
    ("torch", "what", "a tensor", "A tensor is a grid of numbers, like a list, an image of pixels, or many stacked at once."),
    ("torch", "what", "autograd", "Autograd records every operation on a tensor so gradients can flow back automatically."),
    ("torch", "what", "requires_grad", "Write requires_grad equal True on a tensor, and PyTorch tracks it for gradient computation."),
    ("torch", "what", "the zero_grad call", "The zero_grad call empties the gradients from the last step, so each step starts fresh."),
    ("torch", "what", "a Module", "A Module is the base class for models in PyTorch that holds layers and defines the forward pass."),
    ("torch", "what", "the to method", "The to method moves a tensor or model to a device, like the GPU with cuda or back to the CPU."),
    ("torch", "what", "the eval method", "The eval method turns off training behavior like dropout, used before inference or testing."),
    ("torch", "what", "the train method", "The train method turns training behavior like dropout back on before a training run."),
    ("torch", "how", "run PyTorch on the GPU", "Move the model and every batch with the to method using the word cuda, and PyTorch runs on the GPU."),
    ("torch", "how", "save a PyTorch model", "Call torch.save(model.state_dict(), path) to store the learned weights in one file."),
    ("torch", "how", "load a PyTorch model", "Rebuild the model class first, then call load_state_dict with the stored file on the CPU map."),
    ("torch", "why", "we call optimizer.zero_grad in PyTorch", "Gradients from the last step stay stored in the parameters, so without clearing them the new steps add up wrongly."),
    ("torch", "why", "moving data to the GPU matters in PyTorch", "The GPU can only compute on data stored in its own memory, so tensors on the CPU stay idle there."),

    # ---------------- Neural networks ----------------
    ("nn", "what", "a hidden layer", "A hidden layer sits between input and output and turns inputs into new, more useful representations."),
    ("nn", "what", "the loss function", "The loss function measures how wrong the predictions are, so training pushes the loss down."),
    ("nn", "what", "cross-entropy loss", "Cross-entropy loss punishes confident wrong answers more, and we use it for next token prediction."),
    ("nn", "what", "the softmax function", "The softmax function turns raw scores into probabilities that are positive and add up to one."),
    ("nn", "what", "the sigmoid function", "The sigmoid function squashes any number into the range from zero to one."),
    ("nn", "what", "a bias term", "A bias term is an extra learned number added to a layer output, moving it up or down like an intercept."),
    ("nn", "what", "a forward pass", "A forward pass pushes input through every layer to produce an output and, during training, a loss."),
    ("nn", "what", "a learning rate too large", "A learning rate too large makes training jump over good spots, and the loss goes up and down wildly."),
    ("nn", "what", "a learning rate too small", "A learning rate too small makes training crawl, and the loss falls very slowly."),
    ("nn", "what", "warmup", "Warmup slowly raises the learning rate at the start of training to avoid unstable early updates."),
    ("nn", "what", "a weight decay", "Weight decay gently shrinks weights each step, which keeps the model simpler and reduces overfitting."),
    ("nn", "what", "layer normalization", "Layer normalization rescales the values of each layer so training stays stable and faster."),
    ("nn", "what", "a residual connection", "A residual connection adds a layer input back to its output, which helps deep models train well."),
    ("nn", "what", "an epoch", "An epoch is one complete pass through every training example."),
    ("nn", "what", "a validation set", "A validation set is data kept away from training, used to measure generalization while tuning."),
    ("nn", "what", "early stopping", "Early stopping ends training when validation loss stops improving, protecting the best weights."),
    ("nn", "what", "a random seed", "A random seed fixes the random draws, so shuffles and initial weights become reproducible."),
    ("nn", "what", "gradient descent", "Gradient descent updates the weights a small step against the gradient to lower the loss."),
    ("nn", "what", "a mini-batch", "A mini-batch is a small group of examples trained together, cheaper than the full set per step."),
    ("nn", "what", "embedding size", "Embedding size is how many numbers represent one token, and bigger sizes hold richer patterns."),
    ("nn", "what", "inference", "Inference is using a trained model to make predictions instead of training it further."),
    ("nn", "how", "reduce overfitting", "Add dropout, gather more data, make the model smaller, or stop training early at the best validation point."),
    ("nn", "how", "choose the learning rate", "Start from a small value like 0.0003 and watch the loss, then lower it when training is unstable."),

    # ---------------- CNNs / Conv2D / pooling / activations ----------------
    ("cnn", "what", "a convolution", "A convolution slides a small filter over the input and records where the pattern appears."),
    ("cnn", "what", "a kernel", "A kernel is the small grid of learned weights that a convolution slides over the input."),
    ("cnn", "what", "a stride", "A stride is how far the kernel jumps each step, and larger strides give smaller outputs."),
    ("cnn", "what", "padding", "Padding adds zeros around the input border so the kernel can reach every position."),
    ("cnn", "what", "max pooling", "Max pooling keeps the biggest number in each small window, shrinking the image but keeping strong features."),
    ("cnn", "what", "a feature map", "A feature map is the output of one filter showing where its pattern was found."),
    ("cnn", "what", "flattening", "Flattening turns the grid of features into one long list so dense layers can read it."),
    ("cnn", "what", "a channel", "A channel is one depth slice of an image, like red, green, and blue, with three channels total."),
    ("cnn", "why", "CNNs suit images", "Nearby pixels belong together, and a convolution checks local patterns in the same way everywhere."),
    ("cnn", "why", "pooling layers exist", "Pooling shrinks the feature maps, which costs less compute and keeps only the strongest signals."),
    ("cnn", "why", "activations are needed", "Without nonlinearity a stack of layers collapses into one simple linear map and cannot learn curves."),
    ("cnn", "what", "ReLU", "ReLU is an activation function that outputs zero for negative inputs and the input itself otherwise."),
    ("cnn", "what", "leaky ReLU", "Leaky ReLU lets a tiny negative slope through, so negative inputs never fully vanish."),
    ("cnn", "what", "GELU", "GELU is a smooth activation used in Transformers that gates inputs by a soft curve."),
    ("cnn", "what", "the tanh function", "The tanh function squashes numbers into the range from negative one to one."),

    # ---------------- Transformers / attention ----------------
    ("transformer", "what", "the attention mechanism", "The attention mechanism lets each position look back at earlier positions and gather their information by weight."),
    ("transformer", "what", "the query key value idea", "The query asks what I need, the key says what I offer, and the value carries the content to copy."),
    ("transformer", "what", "a causal mask", "A causal mask hides future positions so attention only sees the past, which makes text generation valid."),
    ("transformer", "what", "a position embedding", "A position embedding gives each position its own learned vector, since attention alone ignores order."),
    ("transformer", "what", "a layer stack", "A layer stack applies the same block many times, letting later layers refine earlier ideas."),
    ("transformer", "what", "the feed-forward layer", "The feed-forward layer is a small neural network applied to each token after attention to process it further."),
    ("transformer", "what", "a Transformer", "The Transformer is the architecture behind modern language models; it uses attention to mix token information."),
    ("transformer", "what", "an encoder", "The encoder turns input into a compact numeric representation."),
    ("transformer", "what", "a decoder", "The decoder builds the output sequence step by step and is used for generation."),
    ("transformer", "what", "a context window", "The context window is how many previous tokens the model can use to make its next prediction."),
    ("transformer", "why", "attention uses several heads", "Different heads can track different relations, like grammar, meaning, and simply the shape of the text."),
    ("transformer", "why", "the causal mask matters", "During training every position must predict from the past alone, or the model would peek at the answer."),

    # ---------------- LLM / tokenization ----------------
    ("llm", "what", "a token", "A token is a chunk of text the model reads as one unit; words, word parts, or single characters."),
    ("llm", "what", "tokenization", "Tokenization splits raw text into the units that a model reads, called tokens."),
    ("llm", "what", "BPE", "BPE, byte pair encoding, merges frequent character pairs step by step until common chunks become single tokens."),
    ("llm", "what", "a vocabulary", "The vocabulary is the set of all tokens that a model knows."),
    ("llm", "what", "a special token", "A special token is a reserved symbol with a job, like an end marker for a reply."),
    ("llm", "what", "character-level tokenization", "It treats every single character as a token, so the vocabulary stays tiny but sequences become long."),
    ("llm", "what", "the logits", "The logits are raw scores for every possible next token, and softmax turns them into probabilities."),
    ("llm", "what", "temperature", "The temperature divides the logits before sampling; low values make output steady, high values make it varied."),
    ("llm", "what", "top-k sampling", "Top-k sampling keeps only the k most likely tokens and samples among them."),
    ("llm", "what", "top-p sampling", "Top-p sampling keeps the smallest set of tokens whose probabilities add up to p, and samples among them."),
    ("llm", "what", "greedy decoding", "Greedy decoding always picks the single most likely next token, so output is deterministic."),
    ("llm", "what", "autoregressive generation", "Autoregressive generation predicts the next token from all previous ones, then appends it and repeats."),
    ("llm", "what", "an eos token", "The eos token marks the end of a sequence, so generation can stop cleanly."),
    ("llm", "why", "subword tokens help", "They shrink sequences and share word parts, so rarer words stay inside the known vocabulary."),
    ("llm", "why", "character vocabulary stays small", "A language has few distinct characters, so every unknown word is still spelled with known pieces."),

    # ---------------- ML fundamentals ----------------
    ("ml", "what", "supervised learning", "Supervised learning trains on examples with known answers so the model learns the mapping from input to output."),
    ("ml", "what", "unsupervised learning", "Unsupervised learning finds structure in data without any labeled answers."),
    ("ml", "what", "a feature", "A feature is an input value that describes something about an example."),
    ("ml", "what", "a label", "A label is the answer the model should learn to predict for an example."),
    ("ml", "what", "a training set", "The training set is the data a model learns from directly."),
    ("ml", "what", "a test set", "The test set is held-out data used once at the end to report honest results."),
    ("ml", "what", "regularization", "Regularization adds a small cost for complexity so the model generalizes better."),
    ("ml", "what", "data leakage", "Data leakage lets the model peek at test information during training, which inflates scores."),
    ("ml", "what", "the variance of a model", "High variance means the model fits noise, doing great on seen data and poorly on new data."),
    ("ml", "what", "a hyperparameter", "A hyperparameter is a setting chosen before training, like the learning rate or the layer depth."),
    ("ml", "what", "scaling features", "Scaling means putting features on a similar range so training is not dominated by one large column."),
    ("ml", "what", "the perceptron", "The perceptron is an early simple model that learns a straight boundary between two classes."),
    ("ml", "what", "the Turing test", "The Turing test asks whether a machine conversation feels indistinguishable from a human."),
    ("ml", "why", "we shuffle training data", "Shuffling stops the model from meeting all examples of one kind in a boring, skewed order."),

    # ---------------- English language ----------------
    ("english", "what", "a noun", "A noun is a word for a person, place, thing, or idea, like dog, city, or idea."),
    ("english", "what", "a verb", "A verb shows an action or a state, like run, write, is, or seems."),
    ("english", "what", "an adjective", "An adjective describes a noun, like tall, bright, or careful."),
    ("english", "what", "an adverb", "An adverb modifies a verb or an adjective, like quickly or very."),
    ("english", "what", "a pronoun", "A pronoun stands in for a noun, like he, she, it, or they."),
    ("english", "what", "a subject", "The subject is the person or thing that the sentence is about and that performs the action."),
    ("english", "what", "an object", "The object receives the action of the verb, like the ball in throw the ball."),
    ("english", "what", "a paragraph", "A paragraph is a group of sentences that develop one idea."),
    ("english", "what", "a topic sentence", "The topic sentence states the main idea of a paragraph, usually at the start."),
    ("english", "what", "a plural", "The plural names more than one, and many nouns add s or es to become plural."),
    ("english", "how", "build a clear sentence", "Start with the subject, give its verb, and add details after, keeping each idea placed once."),
    ("english", "how", "improve a run-on sentence", "Split it into two sentences or join the ideas with a comma and a joining word."),

    # ---------------- Debugging / code explanation ----------------
    ("debug", "what", "a syntax error", "A syntax error means the code breaks the language rules, so it cannot run."),
    ("debug", "what", "a runtime error", "A runtime error appears while a program runs, like dividing by zero or wrong typing."),
    ("debug", "what", "a logic error", "A logic error makes the program run but produce wrong results, because the plan itself is off."),
    ("debug", "what", "an index out of range error", "An index out of range means you asked for a position that the sequence does not have."),
    ("debug", "what", "a type error", "A type error means two values cannot do that operation together, like adding 3 to the text 3."),
    ("debug", "what", "the stack trace", "The stack trace prints the chain of function calls at the moment the error occurred, ending at the failing line."),
    ("debug", "what", "print debugging", "It means adding print statements to see what values your code has at each step."),
    ("debug", "how", "fix an off-by-one error", "Check whether your loop should end before the last item or include it, and test the edge positions."),
    ("debug", "how", "fix an indentation error", "Align the blocks from the start of the line and use the same spacing everywhere in a file."),
    ("debug", "how", "read a short error message", "Read the last line for the error type, then jump to the file and line the trace above it mentions."),
    ("debug", "how", "debug a loop that never ends", "Check the variable that the condition depends on, and confirm that something changes it every pass."),
    ("debug", "how", "explain a piece of code", "Say what the input is, what each part changes, and what comes out at the end."),

    # ---------------- General conversation ----------------
    ("general", "what", "a computer program", "A program is a list of steps the computer follows, usually written as code."),
    ("general", "what", "the internet", "The internet is a network that connects computers worldwide so they can share information."),
    ("general", "how", "get better at programming", "Practice small problems, read other code, and fix your failures one step at a time."),
    ("general", "how", "stay organized while coding", "Name things clearly, keep functions small, and test each piece as you build."),
]

FOLLOW_UP_QA = [
    ("My name is {name}", "Nice to meet you, {name}. Ask me anything about programming or science."),
    ("What is my name, do you remember?", "You told me your name is {name}."),
    ("Goodbye for now", "Goodbye, {name}, and come back any time."),
]

FOLLOW_UP_TOPICS = [
    ("I want to learn about {topic}", "A {topic} is a small step toward your goal. Study it by doing tiny examples first."),
    ("And what about using {topic}?", "You can start {topic} with a tiny example and write down the result to see it work."),
    ("Which topic seems easier, loops or {topic}?", "Start with the easier one, build confidence, and then return to the harder topic."),
]


def _load_existing_questions() -> set[str]:
    text = DATA_FILE.read_text(encoding="utf-8")
    out = set()
    for line in text.splitlines():
        if line.startswith("User:"):
            out.add(normalize_question(line[len("User:"):].strip()))
    return out


def build_fact_examples() -> list[Example]:
    """Grouped fact bank to paraphrased question examples."""
    out = []
    for cat, qtype, base, answer in FACTS:
        if qtype == "what":
            out.extend(make_what(cat, base, answer))
        elif qtype == "how":
            out.extend(make_how(cat, base, answer))
        elif qtype == "why":
            for tmpl in WHY_WRAPPERS:
                q = _capitalize(tmpl.format(b=base))
                out.append(Example(q, answer, source="expansion/facts", category=cat))
    return out


def build_arithmetic_examples(seed: int = 42) -> list[Example]:
    """Programmatic arithmetic. Binary ops use the medium band (at least one
    operand < 10); large-band combinations are reserved for the test set."""
    rng = random.Random(seed)
    out: list[Example] = []

    def note(q: str, a: str) -> None:
        out.append(Example(q, a, source="expansion/arithmetic", category="arithmetic"))

    QWHAT = ["What is {q}?", "Can I ask, what is {q}?",
             "Please tell me, what is {q}?", "I was wondering, what is {q}?"]
    QCALC = ["Calculate {q}", "Please calculate {q}"]
    QEQA = ["What does {q} equal?", "What does {q} equal, can you tell?"]

    def pick(qfmt, expr, ans, styles=("what", "what", "calc", "eq")):
        for s in styles:
            if s == "what":
                q = rng.choice(QWHAT).format(q=expr)
            elif s == "calc":
                q = rng.choice(QCALC).format(q=expr)
            else:
                q = rng.choice(QEQA).format(q=expr)
            note(_capitalize(q), ans)

    # Addition: operands 0-9 (medium band)
    for a in range(1, 10):
        for b in range(1, 10):
            if (a + b) % 3 == 0:  # keep a manageable subset, deterministic
                expr = f"{a} + {b}"
                pick(None, expr, f"{a} + {b} is {a + b}.")
    # Subtraction: non-negative, at least one operand < 10
    for a in range(5, 31):
        for b in range(1, min(a, 10)):
            if (a + b) % 4 == 0:
                expr = f"{a} - {b}"
                pick(None, expr, f"{a} - {b} is {a - b}.", styles=("what", "calc"))
    # Multiplication: 2-9 table
    for a in range(2, 10):
        for b in range(2, 10):
            if (a * b) % 2 == 0 or (a + b) % 3 == 0:
                expr = f"{a} times {b}"
                pick(None, expr, f"{a} times {b} is {a * b}.", styles=("what", "eq", "calc"))
    # Division: exact, divisor < 10
    for b in range(2, 10):
        for q_res in range(2, 13):
            a = b * q_res
            if a % 2 == 0 or q_res % 3 == 0:
                expr = f"{a} divided by {b}"
                pick(None, expr, f"{a} divided by {b} is {q_res}.", styles=("what", "calc"))
    # Single-number helpers: squares, halves, doubles, evens (medium band by rule)
    for n in range(4, 21, 2):
        expr = f"the square of {n}"
        note(_capitalize(QWHAT[0].format(q=expr)), f"The square of {n} is {n * n}.")
        expr2 = f"half of {n}"
        note(_capitalize(QWHAT[1].format(q=expr2)), f"Half of {n} is {n // 2}.")
        if n <= 15:
            note(_capitalize(QWHAT[2].format(q=f"double {n}")), f"Double {n} is {n * 2}.")
    for n in range(3, 26):
        if n % 2 == 0:
            note(_capitalize(QCALC[0].format(q=f"if {n} is even or odd")), f"{n} is even.")
        else:
            note(_capitalize(QCALC[0].format(q=f"if {n} is even or odd")), f"{n} is odd.")
    # Percentages (word 'percent', never the percent sign)
    for p in (5, 10, 25, 50):
        for base in range(20, 201, 20):
            if (p + base) % 3 == 0:
                val = p * base // 100
                note(_capitalize(QWHAT[0].format(q=f"{p} percent of {base}")),
                     f"{p} percent of {base} is {val}.")
    return out


def build_sequence_examples(seed: int = 42) -> list[Example]:
    """Number sequences. Train steps 1-9; larger steps held out for eval."""
    rng = random.Random(seed)
    out = []
    QUEST = ["What is the next number: {s}?",
             "Please tell me, what is the next number: {s}?",
             "Can I ask, what is the next number: {s}?"]
    for step in range(1, 10):
        for start in (rng.choice(range(1, 6)), rng.choice(range(2, 8))):
            if step >= 6 and start % 2 == 0:
                continue
            seq = [start]
            for _ in range(3):
                seq.append(seq[-1] + step)
            s_txt = ", ".join(str(x) for x in seq)
            ans_val = seq[-1] + step
            note_q = rng.choice(QUEST).format(s=s_txt)
            out.append(Example(_capitalize(note_q),
                               f"The next number is {ans_val}, because each number goes up by {step}.",
                               source="expansion/sequences", category="arithmetic"))
    return out


def build_program_output_examples(seed: int = 42) -> list[Example]:
    """Held out constants: only small numbers appear so the test set can use
    bigger ones. Program output questions use the print function."""
    rng = random.Random(seed)
    out = []
    T = ["What does print({expr}) show in Python?",
         "In Python, what does print({expr}) output?",
         "Please tell me, what does print({expr}) display?"]
    A = ["It prints {v}.", "The output is {v}.", "It displays {v}."]

    def add(expr: str, value: str) -> None:
        q = rng.choice(T).format(expr=expr)
        a = rng.choice(A).format(v=value)
        out.append(Example(q, a, source="expansion/program_output",
                           category="program_output"))

    # arithmetic inside print (medium band)
    add("2 + 3", "5"); add("8 - 3", "5"); add("3 * 4", "12")
    add("9 - 2", "7"); add("2 * 5", "10"); add("7 + 2", "9")
    # text arithmetic
    add("\"ab\" + \"c\"", "abc"); add("\"hi\" * 2", "hihi")
    # len and indexing
    add("len(\"cat\")", "3"); add("len([1, 2, 3])", "3")
    add("len(\"hello\")", "5"); add("len([])", "0")
    add("[1, 2, 3][0]", "1"); add("[1, 2, 3][2]", "3")
    add("[10, 20, 30][1]", "20")
    # min max sum
    add("min(2, 5)", "2"); add("max(2, 5)", "5"); add("sum([1, 2, 3])", "6")
    add("sum([2, 3])", "5")
    # string methods
    add("\"cat\".upper()", "CAT"); add("\"cat\".title()", "Cat")
    add("\"AB\".lower()", "ab")
    # boolean
    add("3 > 2", "True"); add("2 > 3", "False"); add(  "4 == 4", "True")
    return out


def build_followup_examples() -> list[Example]:
    """Multi-turn style: answers that reference earlier turns."""
    out = []
    names = ["Leo", "Mia", "Max", "Ada"]
    topics = [("a variable", "variable"), ("for loops", "for loops"),
              ("functions", "functions")]
    for name in names:
        for tmpl_q, tmpl_a in FOLLOW_UP_QA:
            q = tmpl_q.format(name=_capitalize(name))
            a = tmpl_a.format(name=name)
            out.append(Example(q, a, source="expansion/followup", category="followup"))
    for topic_long, topic in topics:
        for tmpl_q, tmpl_a in FOLLOW_UP_TOPICS:
            q = tmpl_q.format(topic=topic_long)
            a = tmpl_a.format(topic=topic)
            out.append(Example(q, a, source="expansion/followup", category="followup"))
    return out


def generate_expanded_examples(seed: int = 42) -> list[Example]:
    """All expansion content, joined, deduped against data.txt questions."""
    existing = _load_existing_questions()
    examples = (
        build_fact_examples()
        + build_arithmetic_examples(seed=seed)
        + build_sequence_examples(seed=seed)
        + build_program_output_examples(seed=seed)
        + build_followup_examples()
    )
    counts = Counter(e.category for e in examples)
    kept, seen = [], set()
    dup_stats = Counter()
    for ex in examples:
        qn = normalize_question(ex.question)
        if qn in existing or qn in seen:
            dup_stats["dropped_duplicate"] += 1
            continue
        seen.add(qn)
        kept.append(ex)
    ex_summary = {"kept": len(kept), "dropped": dict(dup_stats),
                  "by_category": dict(counts)}
    Path(PROJECT_DIR / "data" / "expansion_report.json").write_text(
        json.dumps(ex_summary, indent=2), encoding="utf-8")
    return kept


if __name__ == "__main__":
    exs = generate_expanded_examples()
    print(f"kept: {len(exs)}")
    for e in exs[:5]:
        print(e.format(), "\n---")
