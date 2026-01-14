import hljs from '@highlightjs/cdn-assets/es/highlight.js'

function mlir(hljs) {
    var ID = '[\\w\\d_$.]+';
    var PRIMITIVE_TYPES = {
        className: 'type',
        begin: '[x\\b\\s]*(i\\d+|f(16|32|64)|bf16)',
    };

    var SEMI_AFFINE_MAP = {
        className: 'attr',
        begin: '\\([^)>]*\\)\\s*->\\s*\\([^)>]*\\)'
    };
    var LAYOUT_SPECIFICATION = {
        className: 'type',
        variants: [SEMI_AFFINE_MAP]
    };

    return {
        name: 'MLIR',
        keywords:
            'func module ' +
            'br cond_br return',
        contains: [
            PRIMITIVE_TYPES,
            {
                className: 'type',
                begin: '!' + ID,
            },
            hljs.C_LINE_COMMENT_MODE,
            hljs.QUOTE_STRING_MODE,
            {
                className: 'type', begin: '(memref|tensor|vector)<\\b', end: '>',
                keywords: "memref tensor vector",
                contains: [
                    {
                        className: 'number',
                        variants: [
                            { begin: '[*]x' },
                            { begin: '((\\?|\\d+)\\s*x\\s*)+' },
                        ]
                    },
                    'self',
                    PRIMITIVE_TYPES,
                    LAYOUT_SPECIFICATION
                ]
            },
            {
                className: 'keyword', begin: 'affine_map<', end: '>',
                keywords: 'affine_map',
                contains: [
                    SEMI_AFFINE_MAP
                ]
            },
            {
                className: 'title',
                variants: [
                    { begin: '@' + ID },
                    { begin: '@\\d+' },
                ]
            },
            {
                className: 'symbol',
                variants: [
                    { begin: '%' + ID + '([:#]\\d+)?' },
                    { begin: '\\^' + ID },
                    { begin: '#' + ID },
                ]
            },
            hljs.C_NUMBER_MODE
        ]
    };
}

hljs.registerLanguage('mlir', mlir);

export function highlightMLIR(elem) {
    hljs.highlightElement(elem);
}
