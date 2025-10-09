grammar GraphAlg;

program: func*;
func: FUNC name=IDENT params '->' return_type=type block;

params: '(' ')'
      | '(' param (',' param)* ','? ')';
param: name=IDENT ':' type;

type: semiring                #typeScalar
    | MATRIX '<' rows=dim ',' cols=dim ',' semiring '>' #typeMatrix
    | VECTOR '<' rows=dim ',' semiring '>' #typeVector
    ;

dim: ONE_VAL    #dimOne
   | IDENT      #dimSymb
   ;

semiring: BOOL          #semiringBool
        | INT           #semiringInt
        | REAL          #semiringReal
        | TROP_INT      #semiringTropInt
        | TROP_REAL     #semiringTropReal
        | TROP_MAX_INT  #semiringTropMaxInt
        ;

block: '{' stmt* '}';

stmt: base=IDENT mask fill '=' expr ';'                         #stmtAssign
    | base=IDENT '+' '=' expr ';'                               #stmtAccumulate
    | FOR iterVar=IDENT IN range block until?                   #stmtFor
    // Note: a well-formed program should only have a return at the end of a
    // function. We allow parsing it as a regular statement so that we can print
    // a better error message.
    | RETURN expr ';'                                           #stmtReturn
    ;

mask: '<' name=IDENT '>'        #maskMask
    | '<' '!' name=IDENT '>'    #maskComplement
    |                           #maskNone
    ;

fill: '[' ':' ']'           #fillVector
    | '[' ':' ',' ':' ']'   #fillMatrix
    |                       #fillNone
    ;

range: begin=expr ':' end=expr  #rangeConst
     | expr                     #rangeDim
     ;

until: UNTIL expr ';';

expr: '(' expr ')'                                              #exprParens
    // Ordering of these rules matters for precedence because they are
    // left-recursive.
    | expr '.' T                                                #exprTranspose
    | expr '.' NROWS                                            #exprNrows
    | expr '.' NCOLS                                            #exprNcols
    | expr '.' NVALS                                            #exprNvals
    // Note: syntax of scalar multiply is the same as matrix multiplication.
    | lhs=expr binop rhs=expr                                   #exprBinop
    | lhs=expr '(' '.' binop ')' rhs=expr                       #exprEwise
    | lhs=expr '(' '.' fname=IDENT ')' rhs=expr                 #exprEwiseFunc
    // Variables are unambiguous
    | IDENT                                                     #exprVar
    // Other rules with a clear prefix
    | '!' expr                                                  #exprNot
    | '-' expr                                                  #exprNeg
    | MATRIX '<' semiring '>' '(' rows=expr ',' cols=expr ')'   #exprMatrix
    | VECTOR '<' semiring '>' '(' rows=expr ')'                 #exprVector
    | DIAG '(' expr ')'                                         #exprDiag
    | APPLY '(' fname=IDENT ',' expr (',' expr)? ')'            #exprApply
    | SELECT '(' fname=IDENT ',' expr (',' expr)? ')'           #exprSelect
    | TRIL '(' expr ')'                                         #exprTril
    | TRIU '(' expr ')'                                         #exprTriu
    | REDUCE_ROWS '(' expr ')'                                  #exprReduceRows
    | REDUCE_COLS '(' expr ')'                                  #exprReduceCols
    | REDUCE '(' expr ')'                                       #exprReduce
    | CAST '<' semiring '>' '(' expr ')'                        #exprCast
    | semiring '(' literal ')'                                  #exprLiteral
    | ZERO '(' semiring ')'                                     #exprZero
    | ONE '(' semiring ')'                                      #exprOne
    | PICK_ANY '(' expr ')'                                     #exprPickAny
    ;

binop: '+'     #binopAdd
     | '-'     #binopSub
     | '*'     #binopMul
     | '/'     #binopDiv
     | '=='    #binopEq
     | '!='    #binopNe
     | '<'     #binopLt
     | '>'     #binopGt
     | '<='    #binopLe
     | '>='    #binopGe
     ;

literal: intLiteral     #literalInt
       | REAL_VAL       #literalReal
       | TRUE           #literalTrue
       | FALSE          #literalFalse
       ;

intLiteral: ONE_VAL
          | INT_VAL;

// Language keywords
FUNC: 'func';
RETURN: 'return';
FOR: 'for';
IN: 'in';
UNTIL: 'until';
TRUE: 'true';
FALSE: 'false';

// Builtin functions
DIAG: 'diag';
APPLY: 'apply';
SELECT: 'select';
TRIL: 'tril';
TRIU: 'triu';
REDUCE_ROWS: 'reduceRows';
REDUCE_COLS: 'reduceCols';
REDUCE: 'reduce';
CAST: 'cast';
ZERO: 'zero';
ONE: 'one';
PICK_ANY: 'pickAny';

// Semirings
BOOL: 'bool';
INT: 'int';
REAL: 'real';
TROP_INT: 'trop_int';
TROP_REAL: 'trop_real';
TROP_MAX_INT: 'trop_max_int';

// Matrices
MATRIX: 'Matrix';
VECTOR: 'Vector';

// Properties
T: 'T';
NROWS: 'nrows';
NCOLS: 'ncols';
NVALS: 'nvals';

// Whitespace
LINE_COMMENT: '//' ~('\r' | '\n')* -> skip;
WS: [ \t\r\n]+ -> skip;

// Core tokens
IDENT: [a-zA-Z] [a-zA-Z0-9_']*;
ONE_VAL: '1';
INT_VAL: [0-9]+;
REAL_VAL: [0-9]+ '.' [0-9]+;
