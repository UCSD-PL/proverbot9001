# Building Your Own Predictor

Proverbot9001 comes with a bunch of infrastructure around predictors
that handles interfacing with coq and building reports. You can use
this infrastructure to judge the effectiveness of different tactic
predicting techniques.

In this document I'll walk you through creating a basic
predictor. This predictor will predict "intro" whenever it sees a
"forall" in the goal, and it'll predict "auto" otherwise. Pretty
smart, huh?

This tutorial will use the unix shell, and a text editor. Hopefully
you have a text editor of your own you like, as text editor material
won't be covered. For the tutorial, I'll use the `vim` editor, so
wherever you see `vim`, edit the file with the text editor of your
choice. Start by logging in to whatever computer you want to use, and
going to the directory where you put the proverbot9001 code. I'll
assume in this tutorial that it's in your home directory, and called
"proverbot9001".

This tutorial will involve editing code files. If you're using a
remote machine to run Proverbot9001, you can either use an editor on
the remote machine, or edit files in your local copy and use `scp` or
`rsync` to move them to the remote machine.

If you don't know how to use rsync, check
out
[this tutorial](https://www.digitalocean.com/community/tutorials/how-to-use-rsync-to-sync-local-and-remote-directories-on-a-vps)

# Creating The Class

To start, let's create a python file that will hold our predictor.

```bash
vim intro_auto_predictor.py
```

Next, add this to your file:

```python
from tactic_predictor import TacticPredictor

class IntroAutoPredictor(TacticPredictor):
    def __init__(self, options):
        pass
    def predictKTactics(self, in_data, k):
        ...
```

That's a bunch of code! Let's go over it line-by-line

```python
from tactic_predictor import TacticPredictor
```

The first line is an import statement. It opens the file
`tactic_predictor.py` in the current directory, and imports the class
`TacticPredictor`. This is the class that all predictors are derived
from.

```python
class IntroAutoPredictor(TacticPredictor):
```

Next, we're declaring our new predictor class. The class is called
`IntroAutoPredictor`, and it's a kind of `TacticPredictor`. Having all
our predictors derive from `TacticPredictor` allows us to maintain a
consistent interface for predictors, so that we can write the rest of
our code without worrying about which predictor we're using.

```python
    def __init__(self, options):
        pass
```

This is the initializing method. The Proverbot9001 infrastructure will
call this at the beginning of the program, and you can use it to set
up the predictor. For the more complicated predictors, this often
involves loading state using a filename from the `option`
parameter. For this simple predictor however, we don't need to do any
initialization, so we just write `pass` to end the function.

```python
    def predictKTactics(self, in_data, k):
        ...
```

This is the main prediction interface. The report code will call this
function when it wants your predictor to predict something. Often the
reports want a few different predictions at once, so the `k` parameter
tells you how many predictions to produce. The `in_data` parameter is
a dictionary mapping strings to sentences of data. It generally
includes a few entries for `"goal"`, `"hypothesis"`, etc.

Right now we have the function body filled with `...`, because we
haven't implemented it yet.

# Understanding the Input

Since our predictor predicts "intro" when it sees a "forall" in the
goal, we first need to figure out when there is a forall in the
goal. To do that, we'll use the python "regular expression"
library. This library does a lot of stuff, but we'll only need the
`search` function. So at the top of the file, add:

```python
from re import search
```

Then, in the `predictKTactics` function, delete the `...`, and add the
line:

```python
has_forall = search("forall", in_data["goal"])
```

This searches the "goal" input for the string "forall", and returns a
match object if it finds it. If not, it just returns `None`. Match
objects have a lot going on, but since we only care whether the string
is there or not, it's only important that the result is either an
object or `None`.

Next, after that line, add
```python
if has_forall:
    return ["intro."] * k
else:
    return ["auto."] * k
```

This code snippet will execute the second line, `return ["intro."] *
k` if has_forall is an object, and will execute the fourth line,
`return ["auto."] * k` if it's `None`. In general, most values in
python turn into `True` if you give them to an `if`. But `None` turns
into `False`. So values that are either some object, or `None` are
good for using as branch conditions. If you wanted to, you could
instead write:

```python
if has_forall != None:
```

which explicitly checks if has_forall is not `None`. The two lines are
equivalent.

Finally, the lines:

```python
    return ["intro."] * k
```

and

```python
    return ["auto."] * k
```

return the predicted tactics. The `return` keyword means that the
value coming after should be the result of the function.

The expression `["intro."] * k` means produce a list that has `k`
items, and every item is the string "intro.". Similarly for
`["auto."] * k`. These are the predictions produced by your new
predictor.

# Run the Predictor

Finally, let's see how well this predictor works.

Proverbot9001 keeps a list of all known predictors in the file
`predict_tactic.py`. Open that file in your editor:

```bash
vim predict_tactic.py
```

And below the line
```python
import encdecrnn_predictor
```

Add a new line
```python
import intro_auto_predictor
```

And then below the line
```python
    'encdecrnn' : encdecrnn_predictor.EncDecRNNPredictor,
```

Add
```python
    'introauto' : intro_auto_predictor.IntroAutoPredictor,
```

And save the file. That will register your new predictor in the
Proverbot9001 system, with the name "introauto".

Now all you have to do is run a report with the new predictor. Run:

```bash
make FLAGS=--predictor="introauto" report
```

This will make a report like normal, but first set the Makefile
variable `FLAGS` to the string `--predictor="introauto"`. The Makefile
will in turn pass this through to the reports code, which will use the
"introauto" predictor as a result.

Wait for that report to finish, then check it out. How did it do?
Where does it do well? Where does it do badly?
