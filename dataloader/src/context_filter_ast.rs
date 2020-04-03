
pub enum ContextFilterAST {
    And(Vec<ContextFilterAST>),
    Or(Vec<ContextFilterAST>),

    None,
    All,
    GoalArgs,
    HypArgs,
    RelevantLemmaArgs,
    NumericArgs,
    NoSemis,
    Default,

    Tactic(String),
    MaxArgs(i64),
}
