# 1 "quickChickToolLexer.mll"
 
open Lexing
open QuickChickToolParser
open QuickChickToolTypes

(* Function to increase line count in lexbuf *)
let line_incs s lexbuf =
(*  Printf.printf "Read: %s\n" s; *)
  let splits = Str.split_delim (Str.regexp "\n") s in 
  let pos = lexbuf.Lexing.lex_curr_p in
(* Printf.printf "Was in line %d, position %d\n" pos.pos_lnum (pos.pos_cnum - pos.pos_bol); *)
  lexbuf.Lexing.lex_curr_p <- {
    pos with 
      Lexing.pos_lnum = pos.Lexing.pos_lnum + (List.length splits - 1);
      Lexing.pos_bol = if List.length splits > 1 then pos.Lexing.pos_cnum - (String.length (List.hd (List.rev splits))) else pos.Lexing.pos_bol
  }

let python_comment_bit = ref false

# 22 "quickChickToolLexer.ml"
let __ocaml_lex_tables = {
  Lexing.lex_base =
   "\000\000\236\255\237\255\001\000\003\000\003\000\000\000\006\000\
    \000\000\251\255\000\000\000\000\000\000\014\000\000\000\000\000\
    \000\000\000\000\000\000\255\255\002\000\001\000\002\000\000\000\
    \004\000\254\255\001\000\004\000\001\000\000\000\000\000\008\000\
    \015\000\022\000\015\000\253\255\016\000\252\255\250\255\238\255\
    \247\255\001\000\044\000\002\000\242\255\007\000\006\000\027\000\
    \049\000\030\000\015\000\028\000\023\000\025\000\246\255\020\000\
    \036\000\028\000\039\000\025\000\245\255\036\000\043\000\036\000\
    \001\000\040\000\046\000\047\000\049\000\042\000\244\255\045\000\
    \243\255\241\255";
  Lexing.lex_backtrk =
   "\255\255\255\255\255\255\018\000\018\000\018\000\007\000\006\000\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\016\000\015\000\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255";
  Lexing.lex_default =
   "\002\000\000\000\000\000\255\255\255\255\255\255\255\255\255\255\
    \255\255\000\000\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\000\000\255\255\255\255\255\255\255\255\
    \255\255\000\000\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\000\000\255\255\000\000\000\000\000\000\
    \000\000\255\255\255\255\255\255\000\000\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\000\000\255\255\
    \255\255\255\255\255\255\255\255\000\000\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\000\000\255\255\
    \000\000\000\000";
  Lexing.lex_trans =
   "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\013\000\
    \013\000\000\000\000\000\013\000\000\000\000\000\000\000\013\000\
    \013\000\000\000\000\000\013\000\000\000\000\000\000\000\000\000\
    \000\000\007\000\042\000\000\000\000\000\000\000\013\000\009\000\
    \005\000\038\000\004\000\041\000\040\000\006\000\013\000\003\000\
    \008\000\073\000\039\000\000\000\000\000\048\000\048\000\000\000\
    \008\000\048\000\048\000\048\000\000\000\000\000\048\000\000\000\
    \000\000\000\000\000\000\030\000\065\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\048\000\044\000\000\000\000\000\
    \000\000\048\000\000\000\000\000\000\000\000\000\043\000\010\000\
    \000\000\012\000\000\000\043\000\000\000\000\000\000\000\010\000\
    \000\000\012\000\000\000\015\000\024\000\014\000\022\000\028\000\
    \031\000\017\000\027\000\011\000\029\000\032\000\019\000\018\000\
    \023\000\033\000\036\000\011\000\016\000\026\000\021\000\025\000\
    \020\000\034\000\035\000\037\000\061\000\045\000\055\000\047\000\
    \049\000\050\000\045\000\051\000\047\000\052\000\053\000\054\000\
    \056\000\057\000\058\000\059\000\060\000\062\000\063\000\064\000\
    \066\000\046\000\071\000\067\000\069\000\070\000\046\000\068\000\
    \072\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
    \000\000\000\000";
  Lexing.lex_check =
   "\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\000\000\000\000\255\255\255\255\000\000\255\255\007\000\
    \007\000\255\255\255\255\007\000\255\255\255\255\255\255\013\000\
    \013\000\255\255\255\255\013\000\255\255\255\255\255\255\255\255\
    \000\000\006\000\041\000\255\255\255\255\255\255\007\000\007\000\
    \000\000\008\000\000\000\003\000\004\000\005\000\013\000\000\000\
    \007\000\043\000\004\000\255\255\255\255\042\000\042\000\255\255\
    \013\000\042\000\048\000\048\000\255\255\255\255\048\000\255\255\
    \255\255\255\255\255\255\029\000\064\000\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\042\000\042\000\255\255\255\255\
    \255\255\048\000\255\255\255\255\255\255\255\255\042\000\007\000\
    \255\255\007\000\255\255\048\000\255\255\255\255\255\255\013\000\
    \255\255\013\000\255\255\014\000\023\000\012\000\021\000\027\000\
    \030\000\016\000\026\000\007\000\028\000\031\000\018\000\017\000\
    \022\000\031\000\032\000\013\000\015\000\010\000\020\000\024\000\
    \011\000\033\000\034\000\036\000\045\000\042\000\046\000\042\000\
    \047\000\049\000\048\000\050\000\048\000\051\000\052\000\053\000\
    \055\000\056\000\057\000\058\000\059\000\061\000\062\000\063\000\
    \065\000\042\000\067\000\066\000\068\000\069\000\048\000\066\000\
    \071\000\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \000\000\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\255\
    \255\255\255\255";
  Lexing.lex_base_code =
   "";
  Lexing.lex_backtrk_code =
   "";
  Lexing.lex_default_code =
   "";
  Lexing.lex_trans_code =
   "";
  Lexing.lex_check_code =
   "";
  Lexing.lex_code =
   "";
}

let rec lexer lexbuf =
   __ocaml_lex_lexer_rec lexbuf 0
and __ocaml_lex_lexer_rec lexbuf __ocaml_lex_state =
  match Lexing.engine __ocaml_lex_tables __ocaml_lex_state lexbuf with
      | 0 ->
let
# 29 "quickChickToolLexer.mll"
                                      s
# 159 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 29 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartSection s )
# 163 "quickChickToolLexer.ml"

  | 1 ->
let
# 30 "quickChickToolLexer.mll"
                                      s
# 169 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 30 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_Extends s )
# 173 "quickChickToolLexer.ml"

  | 2 ->
let
# 31 "quickChickToolLexer.mll"
                                         s
# 179 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 31 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartQuickChick s )
# 183 "quickChickToolLexer.ml"

  | 3 ->
let
# 32 "quickChickToolLexer.mll"
                                         s
# 189 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 32 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartQuickCheck s )
# 193 "quickChickToolLexer.ml"

  | 4 ->
let
# 34 "quickChickToolLexer.mll"
                      s
# 199 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 34 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartMutTag s )
# 203 "quickChickToolLexer.ml"

  | 5 ->
let
# 35 "quickChickToolLexer.mll"
                                 s
# 209 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 35 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartMutants s )
# 213 "quickChickToolLexer.ml"

  | 6 ->
let
# 36 "quickChickToolLexer.mll"
                     s
# 219 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 36 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartMutant s )
# 223 "quickChickToolLexer.ml"

  | 7 ->
let
# 37 "quickChickToolLexer.mll"
                     s
# 229 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 37 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartComment s )
# 233 "quickChickToolLexer.ml"

  | 8 ->
let
# 39 "quickChickToolLexer.mll"
                    s
# 239 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 39 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_EndComment s )
# 243 "quickChickToolLexer.ml"

  | 9 ->
let
# 42 "quickChickToolLexer.mll"
                                      s
# 249 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 42 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartSection s )
# 253 "quickChickToolLexer.ml"

  | 10 ->
let
# 43 "quickChickToolLexer.mll"
                                      s
# 259 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 43 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_Extends s )
# 263 "quickChickToolLexer.ml"

  | 11 ->
let
# 44 "quickChickToolLexer.mll"
                                         s
# 269 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 44 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartQuickChick s )
# 273 "quickChickToolLexer.ml"

  | 12 ->
let
# 45 "quickChickToolLexer.mll"
                                         s
# 279 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 45 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartQuickCheck s )
# 283 "quickChickToolLexer.ml"

  | 13 ->
let
# 47 "quickChickToolLexer.mll"
                      s
# 289 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 47 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartMutTag s )
# 293 "quickChickToolLexer.ml"

  | 14 ->
let
# 48 "quickChickToolLexer.mll"
                                 s
# 299 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 48 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartMutants s )
# 303 "quickChickToolLexer.ml"

  | 15 ->
let
# 49 "quickChickToolLexer.mll"
                     s
# 309 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 49 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartMutant s )
# 313 "quickChickToolLexer.ml"

  | 16 ->
let
# 50 "quickChickToolLexer.mll"
                     s
# 319 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 50 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_StartComment s )
# 323 "quickChickToolLexer.ml"

  | 17 ->
let
# 52 "quickChickToolLexer.mll"
                    s
# 329 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 52 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_EndComment s )
# 333 "quickChickToolLexer.ml"

  | 18 ->
let
# 55 "quickChickToolLexer.mll"
               s
# 339 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos (lexbuf.Lexing.lex_curr_pos + -1)
and
# 55 "quickChickToolLexer.mll"
                               c
# 344 "quickChickToolLexer.ml"
= Lexing.sub_lexeme_char lexbuf (lexbuf.Lexing.lex_curr_pos + -1) in
# 55 "quickChickToolLexer.mll"
                                            ( line_incs (s^(String.make 1 c)) lexbuf; 
                                              T_Char (s^(String.make 1 c)) )
# 349 "quickChickToolLexer.ml"

  | 19 ->
let
# 57 "quickChickToolLexer.mll"
               s
# 355 "quickChickToolLexer.ml"
= Lexing.sub_lexeme lexbuf lexbuf.Lexing.lex_start_pos lexbuf.Lexing.lex_curr_pos in
# 57 "quickChickToolLexer.mll"
                                            ( line_incs s lexbuf; T_Eof s )
# 359 "quickChickToolLexer.ml"

  | __ocaml_lex_state -> lexbuf.Lexing.refill_buff lexbuf;
      __ocaml_lex_lexer_rec lexbuf __ocaml_lex_state

;;

