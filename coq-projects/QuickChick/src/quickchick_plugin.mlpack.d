src/quickchick_plugin_MLPACK_DEPENDENCIES:=src/error src/genericLib src/genLib src/coqLib src/semLib src/setLib src/unify src/weightmap src/arbitrarySized src/arbitrarySizedST src/sizeUtils src/sized src/sizeSMon src/sizeMon src/sizeCorr src/simplDriver src/checkerSizedST src/sizedProofs src/depDriver src/driver src/quickChick src/tactic_quickchick
src/quickchick_plugin.cmo:$(addsuffix .cmo,$(src/quickchick_plugin_MLPACK_DEPENDENCIES))
src/quickchick_plugin.cmx:$(addsuffix .cmx,$(src/quickchick_plugin_MLPACK_DEPENDENCIES))
src/error.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/genericLib.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/genLib.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/coqLib.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/semLib.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/setLib.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/unify.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/weightmap.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/arbitrarySized.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/arbitrarySizedST.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/sizeUtils.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/sized.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/sizeSMon.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/sizeMon.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/sizeCorr.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/simplDriver.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/checkerSizedST.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/sizedProofs.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/depDriver.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/driver.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/quickChick.cmx : FOR_PACK=-for-pack Quickchick_plugin
src/tactic_quickchick.cmx : FOR_PACK=-for-pack Quickchick_plugin
