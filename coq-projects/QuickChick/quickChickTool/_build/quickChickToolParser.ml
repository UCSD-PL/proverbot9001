type token =
  | T_Char of (string)
  | T_Extends of (string)
  | T_StartSection of (string)
  | T_StartQuickChick of (string)
  | T_StartQuickCheck of (string)
  | T_StartMutTag of (string)
  | T_StartMutant of (string)
  | T_StartMutants of (string)
  | T_StartComment of (string)
  | T_EndComment of (string)
  | T_Eof of (string)

open Parsing;;
let _ = parse_error;;
# 2 "quickChickToolParser.mly"

open Lexing
open Parsing
open QuickChickToolTypes

(*
type node =
    (* Base chunk of text *)
  | Text of string 
    (* Sections: identifier + a bunch of nodes + extend? *)
  | Section of string * node list * string option
    (* Commented out QuickChick call *)
  | QuickChick of string
    (* Mutant: list of +/- idents, base, list of mutants *)
  | Mutant of (bool * string) list * string * string list 
*)

(* Uncomment for more debugging... *)

# 37 "quickChickToolParser.ml"
let yytransl_const = [|
    0|]

let yytransl_block = [|
  257 (* T_Char *);
  258 (* T_Extends *);
  259 (* T_StartSection *);
  260 (* T_StartQuickChick *);
  261 (* T_StartQuickCheck *);
  262 (* T_StartMutTag *);
  263 (* T_StartMutant *);
  264 (* T_StartMutants *);
  265 (* T_StartComment *);
  266 (* T_EndComment *);
  267 (* T_Eof *);
    0|]

let yylhs = "\255\255\
\001\000\001\000\001\000\010\000\004\000\004\000\005\000\005\000\
\005\000\005\000\005\000\005\000\008\000\008\000\006\000\006\000\
\011\000\011\000\007\000\007\000\003\000\003\000\002\000\009\000\
\009\000\000\000"

let yylen = "\002\000\
\002\000\003\000\002\000\001\000\000\000\002\000\001\000\003\000\
\003\000\002\000\003\000\003\000\001\000\002\000\001\000\002\000\
\004\000\001\000\003\000\001\000\001\000\002\000\005\000\000\000\
\003\000\002\000"

let yydefred = "\000\000\
\000\000\000\000\000\000\007\000\000\000\000\000\000\000\000\000\
\026\000\004\000\000\000\000\000\003\000\000\000\000\000\000\000\
\000\000\000\000\020\000\010\000\018\000\000\000\000\000\000\000\
\006\000\000\000\001\000\000\000\000\000\014\000\008\000\009\000\
\000\000\000\000\011\000\016\000\012\000\000\000\022\000\002\000\
\000\000\019\000\000\000\017\000\000\000\000\000\000\000\023\000\
\025\000"

let yydgoto = "\002\000\
\009\000\028\000\029\000\010\000\011\000\020\000\021\000\015\000\
\046\000\012\000\023\000"

let yysindex = "\006\000\
\042\255\000\000\003\255\000\000\029\255\029\255\053\255\044\255\
\000\000\000\000\044\255\022\255\000\000\029\255\025\255\028\255\
\029\255\029\255\000\000\000\000\000\000\049\255\049\255\030\255\
\000\000\029\255\000\000\033\255\047\255\000\000\000\000\000\000\
\034\255\055\255\000\000\000\000\000\000\057\255\000\000\000\000\
\253\254\000\000\066\255\000\000\029\255\044\255\059\255\000\000\
\000\000"

let yyrindex = "\000\000\
\026\255\000\000\000\000\000\000\000\000\000\000\000\000\060\255\
\000\000\000\000\255\254\000\000\000\000\056\255\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\012\255\000\000\
\000\000\000\000\000\000\061\255\000\000\000\000\000\000\000\000\
\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\
\000\000\000\000\023\255\000\000\000\000\026\255\000\000\000\000\
\000\000"

let yygindex = "\000\000\
\000\000\000\000\043\000\251\255\000\000\252\255\032\000\250\255\
\000\000\000\000\000\000"

let yytablesize = 73
let yytable = "\016\000\
\022\000\005\000\024\000\018\000\019\000\025\000\001\000\030\000\
\005\000\005\000\033\000\034\000\015\000\013\000\015\000\015\000\
\015\000\035\000\036\000\038\000\015\000\015\000\015\000\024\000\
\026\000\024\000\024\000\024\000\005\000\014\000\024\000\024\000\
\027\000\024\000\031\000\026\000\005\000\032\000\047\000\037\000\
\048\000\003\000\004\000\041\000\004\000\005\000\006\000\005\000\
\006\000\007\000\008\000\007\000\008\000\014\000\017\000\018\000\
\019\000\040\000\017\000\018\000\019\000\013\000\013\000\013\000\
\042\000\013\000\043\000\045\000\049\000\005\000\039\000\021\000\
\044\000"

let yycheck = "\006\000\
\007\000\003\001\008\000\007\001\008\001\011\000\001\000\014\000\
\010\001\011\001\017\000\018\000\001\001\011\001\003\001\004\001\
\005\001\022\000\023\000\026\000\009\001\010\001\011\001\001\001\
\003\001\003\001\004\001\005\001\003\001\001\001\008\001\009\001\
\011\001\011\001\010\001\003\001\011\001\010\001\045\000\010\001\
\046\000\000\001\001\001\010\001\001\001\004\001\005\001\004\001\
\005\001\008\001\009\001\008\001\009\001\001\001\006\001\007\001\
\008\001\011\001\006\001\007\001\008\001\006\001\007\001\008\001\
\010\001\010\001\010\001\002\001\010\001\010\001\028\000\011\001\
\041\000"

let yynames_const = "\
  "

let yynames_block = "\
  T_Char\000\
  T_Extends\000\
  T_StartSection\000\
  T_StartQuickChick\000\
  T_StartQuickCheck\000\
  T_StartMutTag\000\
  T_StartMutant\000\
  T_StartMutants\000\
  T_StartComment\000\
  T_EndComment\000\
  T_Eof\000\
  "

let yyact = [|
  (fun _ -> failwith "parser")
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'default_section) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 49 "quickChickToolParser.mly"
                                            ( [_1] )
# 152 "quickChickToolParser.ml"
               : QuickChickToolTypes.section list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : 'default_section) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : QuickChickToolTypes.section list) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 50 "quickChickToolParser.mly"
                                                     ( _1 :: _2 )
# 161 "quickChickToolParser.ml"
               : QuickChickToolTypes.section list))
; (fun __caml_parser_env ->
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 51 "quickChickToolParser.mly"
                                  ( 
                        let pos = Parsing.symbol_start_pos () in
                        failwith (Printf.sprintf "Error in line %d, position %d" 
                                                 pos.pos_lnum (pos.pos_cnum - pos.pos_bol)) )
# 171 "quickChickToolParser.ml"
               : QuickChickToolTypes.section list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : QuickChickToolTypes.node list) in
    Obj.repr(
# 57 "quickChickToolParser.mly"
                        ( { sec_begin = "" 
                          ; sec_name  = ""
                          ; sec_end   = ""
                          ; sec_extends = None
                          ; sec_nodes = _1 }  )
# 182 "quickChickToolParser.ml"
               : 'default_section))
; (fun __caml_parser_env ->
    Obj.repr(
# 63 "quickChickToolParser.mly"
                      ( [] )
# 188 "quickChickToolParser.ml"
               : QuickChickToolTypes.node list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : QuickChickToolTypes.node) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : QuickChickToolTypes.node list) in
    Obj.repr(
# 64 "quickChickToolParser.mly"
                                                       ( _1 :: _2 )
# 196 "quickChickToolParser.ml"
               : QuickChickToolTypes.node list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 67 "quickChickToolParser.mly"
                            (  Text _1 )
# 203 "quickChickToolParser.ml"
               : QuickChickToolTypes.node))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : string list) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 69 "quickChickToolParser.mly"
                            ( QuickChick { qc_begin = _1; qc_body = String.concat "" _2; qc_end = _3 } )
# 212 "quickChickToolParser.ml"
               : QuickChickToolTypes.node))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : string list) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 71 "quickChickToolParser.mly"
                            ( QuickChick { qc_begin = _1; qc_body = String.concat "" _2; qc_end = _3 } )
# 221 "quickChickToolParser.ml"
               : QuickChickToolTypes.node))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : mutant list) in
    Obj.repr(
# 73 "quickChickToolParser.mly"
                            ( Mutants { ms_begin = _1; ms_base = ""; ms_mutants = _2 } )
# 229 "quickChickToolParser.ml"
               : QuickChickToolTypes.node))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : string list) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : mutant list) in
    Obj.repr(
# 75 "quickChickToolParser.mly"
                            ( Mutants { ms_begin = _1; ms_base = String.concat "" _2; ms_mutants = _3 } )
# 238 "quickChickToolParser.ml"
               : QuickChickToolTypes.node))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : QuickChickToolTypes.node list) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 77 "quickChickToolParser.mly"
                            ( Text (Printf.sprintf "%s%s%s" _1 (String.concat "" (List.map output_node _2)) _3) )
# 247 "quickChickToolParser.ml"
               : QuickChickToolTypes.node))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 79 "quickChickToolParser.mly"
                             ( [_1] )
# 254 "quickChickToolParser.ml"
               : string list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : string list) in
    Obj.repr(
# 80 "quickChickToolParser.mly"
                                    ( _1 :: _2 )
# 262 "quickChickToolParser.ml"
               : string list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : 'mutant_tag) in
    Obj.repr(
# 87 "quickChickToolParser.mly"
                                 ( [_1] )
# 269 "quickChickToolParser.ml"
               : mutant list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : 'mutant_tag) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : mutant list) in
    Obj.repr(
# 88 "quickChickToolParser.mly"
                                         ( _1 :: _2 )
# 277 "quickChickToolParser.ml"
               : mutant list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 3 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 2 : string list) in
    let _3 = (Parsing.peek_val __caml_parser_env 1 : string) in
    let _4 = (Parsing.peek_val __caml_parser_env 0 : mutant) in
    Obj.repr(
# 91 "quickChickToolParser.mly"
                        ( let m = _4 in {m with mut_info = {m.mut_info with tag = Some (String.concat "" _2)}} )
# 287 "quickChickToolParser.ml"
               : 'mutant_tag))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : mutant) in
    Obj.repr(
# 92 "quickChickToolParser.mly"
                             ( _1 )
# 294 "quickChickToolParser.ml"
               : 'mutant_tag))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : string list) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 94 "quickChickToolParser.mly"
                                                      ( let pos = Parsing.symbol_start_pos () in
                                                        { mut_info = { file_name = pos.pos_fname
                                                                     ; line_number = pos.pos_lnum 
                                                                     ; tag = None }
                                                        ; mut_begin = _1 ; mut_body = String.concat "" _2 ; mut_end = _3 } )
# 307 "quickChickToolParser.ml"
               : mutant))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 100 "quickChickToolParser.mly"
                        ( let pos = Parsing.symbol_start_pos () in
                          { mut_info = { file_name   = pos.pos_fname
                                       ; line_number = pos.pos_lnum 
                                       ; tag = None }
                          ; mut_begin = "(*" ; mut_body = "" ; mut_end = "*)" } )
# 318 "quickChickToolParser.ml"
               : mutant))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 0 : QuickChickToolTypes.section) in
    Obj.repr(
# 106 "quickChickToolParser.mly"
                              ( [_1] )
# 325 "quickChickToolParser.ml"
               : QuickChickToolTypes.section list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 1 : QuickChickToolTypes.section) in
    let _2 = (Parsing.peek_val __caml_parser_env 0 : QuickChickToolTypes.section list) in
    Obj.repr(
# 107 "quickChickToolParser.mly"
                                       ( _1 :: _2 )
# 333 "quickChickToolParser.ml"
               : QuickChickToolTypes.section list))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 4 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 3 : string list) in
    let _3 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _4 = (Parsing.peek_val __caml_parser_env 1 : extend option) in
    let _5 = (Parsing.peek_val __caml_parser_env 0 : QuickChickToolTypes.node list) in
    Obj.repr(
# 110 "quickChickToolParser.mly"
                        ( { sec_begin   = _1
                          ; sec_name    = String.concat "" _2
                          ; sec_end     = _3
                          ; sec_extends = _4
                          ; sec_nodes   = _5 } )
# 348 "quickChickToolParser.ml"
               : QuickChickToolTypes.section))
; (fun __caml_parser_env ->
    Obj.repr(
# 116 "quickChickToolParser.mly"
                      ( None )
# 354 "quickChickToolParser.ml"
               : extend option))
; (fun __caml_parser_env ->
    let _1 = (Parsing.peek_val __caml_parser_env 2 : string) in
    let _2 = (Parsing.peek_val __caml_parser_env 1 : string list) in
    let _3 = (Parsing.peek_val __caml_parser_env 0 : string) in
    Obj.repr(
# 117 "quickChickToolParser.mly"
                                                  ( Some { ext_begin = _1 ; ext_extends = _2 ; ext_end = _3 } )
# 363 "quickChickToolParser.ml"
               : extend option))
(* Entry program *)
; (fun __caml_parser_env -> raise (Parsing.YYexit (Parsing.peek_val __caml_parser_env 0)))
|]
let yytables =
  { Parsing.actions=yyact;
    Parsing.transl_const=yytransl_const;
    Parsing.transl_block=yytransl_block;
    Parsing.lhs=yylhs;
    Parsing.len=yylen;
    Parsing.defred=yydefred;
    Parsing.dgoto=yydgoto;
    Parsing.sindex=yysindex;
    Parsing.rindex=yyrindex;
    Parsing.gindex=yygindex;
    Parsing.tablesize=yytablesize;
    Parsing.table=yytable;
    Parsing.check=yycheck;
    Parsing.error_function=parse_error;
    Parsing.names_const=yynames_const;
    Parsing.names_block=yynames_block }
let program (lexfun : Lexing.lexbuf -> token) (lexbuf : Lexing.lexbuf) =
   (Parsing.yyparse yytables 1 lexfun lexbuf : QuickChickToolTypes.section list)
