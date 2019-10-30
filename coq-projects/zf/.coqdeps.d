./src/ZFrelations.vo ./src/ZFrelations.glob ./src/ZFrelations.v.beautified: ./src/ZFrelations.v ./src/axs_fundation.vo
./src/ZFrelations.vio: ./src/ZFrelations.v ./src/axs_fundation.vio
./src/nothing.vo ./src/nothing.glob ./src/nothing.v.beautified: ./src/nothing.v
./src/nothing.vio: ./src/nothing.v
./src/axs_remplacement.vo ./src/axs_remplacement.glob ./src/axs_remplacement.v.beautified: ./src/axs_remplacement.v ./src/axs_comprehension.vo
./src/axs_remplacement.vio: ./src/axs_remplacement.v ./src/axs_comprehension.vio
./src/MSetBasis.vo ./src/MSetBasis.glob ./src/MSetBasis.v.beautified: ./src/MSetBasis.v ./src/ZFrelations.vo
./src/MSetBasis.vio: ./src/MSetBasis.v ./src/ZFrelations.vio
./src/useful.vo ./src/useful.glob ./src/useful.v.beautified: ./src/useful.v ./src/nothing.vo
./src/useful.vio: ./src/useful.v ./src/nothing.vio
./src/axs_paire.vo ./src/axs_paire.glob ./src/axs_paire.v.beautified: ./src/axs_paire.v ./src/axs_extensionnalite.vo
./src/axs_paire.vio: ./src/axs_paire.v ./src/axs_extensionnalite.vio
./src/ZFbasis.vo ./src/ZFbasis.glob ./src/ZFbasis.v.beautified: ./src/ZFbasis.v ./src/useful.vo
./src/ZFbasis.vio: ./src/ZFbasis.v ./src/useful.vio
./src/axs_extensionnalite.vo ./src/axs_extensionnalite.glob ./src/axs_extensionnalite.v.beautified: ./src/axs_extensionnalite.v ./src/ZFbasis.vo
./src/axs_extensionnalite.vio: ./src/axs_extensionnalite.v ./src/ZFbasis.vio
./src/applications.vo ./src/applications.glob ./src/applications.v.beautified: ./src/applications.v ./src/couples.vo
./src/applications.vio: ./src/applications.v ./src/couples.vio
./src/axs_parties.vo ./src/axs_parties.glob ./src/axs_parties.v.beautified: ./src/axs_parties.v ./src/axs_reunion.vo
./src/axs_parties.vio: ./src/axs_parties.v ./src/axs_reunion.vio
./src/axs_reunion.vo ./src/axs_reunion.glob ./src/axs_reunion.v.beautified: ./src/axs_reunion.v ./src/axs_paire.vo
./src/axs_reunion.vio: ./src/axs_reunion.v ./src/axs_paire.vio
./src/axs_choice.vo ./src/axs_choice.glob ./src/axs_choice.v.beautified: ./src/axs_choice.v ./src/applications.vo
./src/axs_choice.vio: ./src/axs_choice.v ./src/applications.vio
./src/couples.vo ./src/couples.glob ./src/couples.v.beautified: ./src/couples.v ./src/axs_remplacement.vo
./src/couples.vio: ./src/couples.v ./src/axs_remplacement.vio
./src/axs_comprehension.vo ./src/axs_comprehension.glob ./src/axs_comprehension.v.beautified: ./src/axs_comprehension.v ./src/axs_parties.vo
./src/axs_comprehension.vio: ./src/axs_comprehension.v ./src/axs_parties.vio
./src/axs_fundation.vo ./src/axs_fundation.glob ./src/axs_fundation.v.beautified: ./src/axs_fundation.v ./src/axs_choice.vo
./src/axs_fundation.vio: ./src/axs_fundation.v ./src/axs_choice.vio
