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

val program :
  (Lexing.lexbuf  -> token) -> Lexing.lexbuf -> QuickChickToolTypes.section list
