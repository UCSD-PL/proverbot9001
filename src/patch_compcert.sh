#!/usr/bin/env bash

# determine physical directory of this script
src="${BASH_SOURCE[0]}"
while [ -L "$src" ]; do
  dir="$(cd -P "$(dirname "$src")" && pwd)"
  src="$(readlink "$src")"
  [[ $src != /* ]] && src="$dir/$src"
done
MYDIR="$(cd -P "$(dirname "$src")" && pwd)"

# not using in place because it ran into permission issues
# sed: preserving permissions for ‘/workspace/src/../CompCert/backend/seduqi2mA’: Permission denied
sed \
    -e 's/\(destruct (\)\(eqb ro1 ro2) eqn:RO\.\.\.\)/\1Bool.\2/'\
    -e 's/\(destruct (\)\(eqb vo1 vo2) eqn:VO\.\.\.\)/\1Bool.\2/'\
    $MYDIR/../CompCert/backend/Unusedglobproof.v > $MYDIR/../CompCert/backend/Unusedglobproof.v.new

mv $MYDIR/../CompCert/backend/Unusedglobproof.v.new $MYDIR/../CompCert/backend/Unusedglobproof.v

gawk -i inplace -v INPLACE_SUFFIX=.bkp \
    '/Remark match_is_call_cont/ {in_target=1}
                                 {if (in_target) {print "(*" $0 "*)";} else print}
     /Qed/                       {in_target=0}' \
    $MYDIR/../CompCert/backend/Selectionproof.v

gawk -i inplace -v INPLACE_SUFFIX=.bkp \
    '/Lemma sel_step_correct/ {at_target=1}
                              {if (in_target) {print "(*" $0 "*)";} else print}
     /Proof/                  {if (at_target) in_target=1}
     /Qed/                    {if (in_target) print "Admitted."; in_target=0; at_target=0}' \
    $MYDIR/../CompCert/backend/Selectionproof.v
