
import re
import datetime
import fileinput

def main() -> None:
    total_time = datetime.timedelta(0)
    max_time = datetime.timedelta(0)
    num_lines = 0
    for line in fileinput.input():
        tag, seperator, timedelta = line.partition(":")
        if re.match("vernac", tag):
            # This is a label for how long it took to run the vernac
            continue
        if re.match("Orig.*", tag):
            # This is the time it took to run the original proof
            continue

        theorem_time = parseTimeDelta(timedelta)

        if theorem_time > max_time:
            max_time = theorem_time
        total_time += theorem_time
        num_lines += 1

    print(f"Average theorem search time: {total_time/num_lines}")
    print(f"Max theorem search time: {max_time}")

def parseTimeDelta(s):
    """Create timedelta object representing time delta
       expressed in a string

    Takes a string in the format produced by calling str() on
    a python timedelta object and returns a timedelta instance
    that would produce that string.

    Acceptable format is: "HH:MM:SS.MMMMMM".
    """
    match = re.match(
            r'\s*(?P<hours>\d+):(?P<minutes>\d+):'
            r'(?P<seconds>\d+)\.(?P<milliseconds>\d+)',
            s)
    assert match, f"String {s} didn't match"
    d = match.groupdict(0)
    return datetime.timedelta(**dict(( (key, int(value))
                                       for key, value in d.items() )))

if __name__ == "__main__":
    main()
