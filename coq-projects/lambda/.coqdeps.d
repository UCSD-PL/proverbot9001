Test.vo Test.glob Test.v.beautified: Test.v
Test.vio: Test.v
Terms.vo Terms.glob Terms.v.beautified: Terms.v Test.vo
Terms.vio: Terms.v Test.vio
Substitution.vo Substitution.glob Substitution.v.beautified: Substitution.v Terms.vo Reduction.vo Redexes.vo Test.vo
Substitution.vio: Substitution.v Terms.vio Reduction.vio Redexes.vio Test.vio
Simulation.vo Simulation.glob Simulation.v.beautified: Simulation.v Terms.vo Reduction.vo Redexes.vo Test.vo Marks.vo Substitution.vo Residuals.vo
Simulation.vio: Simulation.v Terms.vio Reduction.vio Redexes.vio Test.vio Marks.vio Substitution.vio Residuals.vio
Residuals.vo Residuals.glob Residuals.v.beautified: Residuals.v Terms.vo Reduction.vo Redexes.vo Test.vo Substitution.vo
Residuals.vio: Residuals.v Terms.vio Reduction.vio Redexes.vio Test.vio Substitution.vio
Reduction.vo Reduction.glob Reduction.v.beautified: Reduction.v Test.vo Terms.vo
Reduction.vio: Reduction.v Test.vio Terms.vio
Redexes.vo Redexes.glob Redexes.v.beautified: Redexes.v Test.vo Terms.vo Reduction.vo
Redexes.vio: Redexes.v Test.vio Terms.vio Reduction.vio
Marks.vo Marks.glob Marks.v.beautified: Marks.v Terms.vo Reduction.vo Redexes.vo Test.vo
Marks.vio: Marks.v Terms.vio Reduction.vio Redexes.vio Test.vio
Lambda.vo Lambda.glob Lambda.v.beautified: Lambda.v Terms.vo Reduction.vo Redexes.vo Residuals.vo Cube.vo Simulation.vo Confluence.vo Conversion.vo
Lambda.vio: Lambda.v Terms.vio Reduction.vio Redexes.vio Residuals.vio Cube.vio Simulation.vio Confluence.vio Conversion.vio
Cube.vo Cube.glob Cube.v.beautified: Cube.v Terms.vo Reduction.vo Redexes.vo Test.vo Substitution.vo Residuals.vo
Cube.vio: Cube.v Terms.vio Reduction.vio Redexes.vio Test.vio Substitution.vio Residuals.vio
Confluence.vo Confluence.glob Confluence.v.beautified: Confluence.v Terms.vo Reduction.vo Redexes.vo Test.vo Marks.vo Substitution.vo Residuals.vo Simulation.vo Cube.vo
Confluence.vio: Confluence.v Terms.vio Reduction.vio Redexes.vio Test.vio Marks.vio Substitution.vio Residuals.vio Simulation.vio Cube.vio
Conversion.vo Conversion.glob Conversion.v.beautified: Conversion.v Terms.vo Reduction.vo Confluence.vo
Conversion.vio: Conversion.v Terms.vio Reduction.vio Confluence.vio
