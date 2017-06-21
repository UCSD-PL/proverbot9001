{ nixpkgs ? import <nixpkgs> {} }:
nixpkgs.stdenv.mkDerivation {
  name = "proverbot9001";
  buildInputs =
    (with nixpkgs; [
      opam
      python3
    ]) ++
    (with nixpkgs.ocamlPackages_4_03; [
      # Coq:
      camlp5_6_strict
      findlib
      ocaml
      # CoqIDE:
      lablgtk
      # SerAPI:
      camlp4
      cmdliner
      ocamlbuild
      ppx_deriving
      ppx_import
      ppx_sexp_conv
      sexplib
      # CompCert
      menhir
    ]) ++
#     (with nixpkgs.ocamlPackages_4_03.janeStreet; [
#       ppx_sexp_conv
#       sexplib
#     ]) ++
    (with nixpkgs.pythonPackages; [
      sexpdata
    ])
  ;
  nativeBuildInputs = (with nixpkgs; [
  ]);
  shellHook = ''
    export NIXSHELL="$NIXSHELL\[proverbot9001\]"
    export SSL_CERT_FILE="/etc/ssl/certs/ca-bundle.crt"
    export SERAPI_COQ_HOME="/home/ptival/proverbot9001/coq/"
  '';
}
