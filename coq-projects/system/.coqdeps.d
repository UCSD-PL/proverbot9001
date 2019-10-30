src/Computation.vo src/Computation.glob src/Computation.v.beautified: src/Computation.v src/Events.vo
src/Computation.vio: src/Computation.v src/Events.vio
src/Events.vo src/Events.glob src/Events.v.beautified: src/Events.v
src/Events.vio: src/Events.v
src/Extraction.vo src/Extraction.glob src/Extraction.v.beautified: src/Extraction.v src/Computation.vo src/Events.vo src/Run.vo
src/Extraction.vio: src/Extraction.v src/Computation.vio src/Events.vio src/Run.vio
src/Run.vo src/Run.glob src/Run.v.beautified: src/Run.v src/Computation.vo src/Events.vo
src/Run.vio: src/Run.v src/Computation.vio src/Events.vio
src/StdLib.vo src/StdLib.glob src/StdLib.v.beautified: src/StdLib.v src/Computation.vo src/Events.vo
src/StdLib.vio: src/StdLib.v src/Computation.vio src/Events.vio
src/Test.vo src/Test.glob src/Test.v.beautified: src/Test.v src/Computation.vo src/Events.vo src/Extraction.vo src/StdLib.vo
src/Test.vio: src/Test.v src/Computation.vio src/Events.vio src/Extraction.vio src/StdLib.vio
