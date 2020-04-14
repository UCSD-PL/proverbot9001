/* *********************************************************************** */
//
//    This file is part of Proverbot9001.
//
//    Proverbot9001 is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    Proverbot9001 is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
//
//    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
//
/* *********************************************************************** */

pub fn split_to_next_matching_paren_or_space<'a>(haystack: &'a str) -> (&'a str, &'a str) {
    let mut depth = 0;
    let mut curpos = 0;
    while curpos < haystack.len() + 1 {
        let next_open = haystack[curpos..]
            .find("(")
            .map(|pos| pos + curpos)
            .unwrap_or(haystack.len() + 1);
        let next_close = haystack[curpos..]
            .find(")")
            .map(|pos| pos + curpos)
            .unwrap_or(haystack.len() + 1);
        let next_split = haystack[curpos..]
            .find(|c| char::is_whitespace(c))
            .map(|pos| pos + curpos)
            .unwrap_or(haystack.len() + 1);
        if next_open < next_close && next_open < next_split {
            depth += 1;
            curpos = next_open + 1;
        } else if next_close < next_open && next_close < next_split {
            depth -= 1;
            curpos = next_close + 1;
        } else if next_split < next_open && next_split < next_close {
            if depth == 0 {
                return (haystack[..next_split].trim(), haystack[next_split..].trim());
            } else {
                curpos = next_split + 1;
            }
        } else if
            next_split == haystack.len() + 1 &&
            next_open == haystack.len() + 1 &&
            next_close == haystack.len() + 1 {
            return (haystack, "");
        } else {
            unimplemented!(
                "Ahhhh: {}, {}, {}, {}, {}",
                next_open,
                next_close,
                next_split,
                haystack.len(),
                haystack
            );
        }
    }
    unimplemented!("Ahhhh: {}", haystack);
}

pub fn split_to_next_pat_outside_parens<'a>(
    haystack: &'a str,
    splitpat: &str,
) -> Option<(&'a str, &'a str)> {
    let mut depth = 0;
    let mut cur_pos = 0;
    macro_rules! lookup {
        ($pat:expr) => {
            haystack[cur_pos..]
                .find($pat)
                .map(|pos| pos + cur_pos)
                .unwrap_or(haystack.len())
        };
    };
    while cur_pos < haystack.len() {
        let next_open = lookup!("(");
        let next_close = lookup!(")");
        let next_split = if depth == 0 {
            lookup!(splitpat)
        } else {
            haystack.len()
        };
        if next_open < next_close && next_open < next_split {
            cur_pos = next_open + 1;
            depth += 1;
        } else if next_close < next_open && next_close < next_split {
            assert!(depth > 0);
            cur_pos = next_close + 1;
            depth -= 1;
        } else if next_split < next_open && next_split < next_close {
            return Some((&haystack[..next_split], &haystack[next_split+1..]));
        } else {
            return None;
        }
    }
    None
}
