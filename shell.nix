{ nixpkgs ? import <nixpkgs> {} }:
nixpkgs.stdenv.mkDerivation {
  name = "proverbot9001";
  buildInputs = (with nixpkgs; [
    python3
  ] ++ (with ocamlPackages_4_02; [
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
      opam
      ppx_import
      ppx_sexp_conv
      sexplib
    ])
    ++ (with pythonPackages; [
      sexpdata
    ])
  );
  nativeBuildInputs = (with nixpkgs; [
  ]);
  shellHook = ''
    export NIXSHELL="$NIXSHELL\[proverbot9001\]"
    export SSL_CERT_FILE="/etc/ssl/certs/ca-bundle.crt"
    export SERAPI_COQ_HOME="/home/ptival/proverbot9001/coq/"
  '';
}
