##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################
import torch
import argparse
from pathlib_revised import Path2

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct weights for reinforced features polyarg")
    parser.add_argument("fpa_weights")
    parser.add_argument("q_weights")
    parser.add_argument("out_weights", type=Path2)
    args = parser.parse_args()

    fpa_name, fpa_saved = torch.load(args.fpa_weights)
    assert fpa_name == "polyarg", "Weights aren't  for an FPA predictor!"
    fpa_args, fpa_up_args, fpa_meta, fpa_state = \
        fpa_saved

    q_name, *q_saved = torch.load(args.q_weights)
    assert q_name == "features evaluator"
    q_args, q_up_args, q_meta, q_state = q_saved

    with args.out_weights.open('wb') as f:
        torch.save(("refpa", (fpa_args, fpa_up_args, (fpa_meta, q_meta),
                             (fpa_state, q_state))),
                   f)

    pass


if __name__ == "__main__":
    main()
