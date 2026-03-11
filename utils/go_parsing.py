import sys
import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

def parse_go_concurrency(code: str | None, file_path : str | None = None):
    # Initialize the Tree-sitter parser for Go
    GO_LANGUAGE = Language(tsgo.language())
    parser = Parser(GO_LANGUAGE)

    if code is None:
        with open(file_path, 'rb') as f:
            tree = parser.parse(f.read())
    else:
        tree = parser.parse(bytes(code, "utf8"))

    root_node = tree.root_node
    results = []

    def traverse(node, current_scope="global"):
        # Update scope if we enter a function or method
        if node.type in ['function_declaration', 'method_declaration']:
            name_node = node.child_by_field_name('name')
            if name_node:
                current_scope = name_node.text.decode('utf8')
        elif node.type == 'func_literal':
            current_scope = f"{current_scope} -> anonymous_func"

        # 1. Goroutines
        if node.type == 'go_statement':
            results.append({
                'name': 'go',
                'scope': current_scope,
                'type': 'goroutine'
            })

        # 2. Select statements
        elif node.type == 'select_statement':
            results.append({
                'name': 'select',
                'scope': current_scope,
                'type': 'select'
            })

        # 3. Channel creation (e.g., ch := make(chan int))
        elif node.type == 'short_var_declaration':
            left = node.child_by_field_name('left')
            right = node.child_by_field_name('right')
            if left and right:
                # Basic check for 'make(chan'
                if b'make(chan' in right.text:
                    results.append({
                        'name': left.text.decode('utf8'),
                        'scope': current_scope,
                        'type': 'chan'
                    })

        # 4. Explicit sync variable declarations (e.g., var m sync.Mutex)
        elif node.type == 'var_declaration':
            for child in node.children:
                if child.type == 'var_spec':
                    name_node = child.child_by_field_name('name')
                    type_node = child.child_by_field_name('type')
                    if name_node and type_node:
                        type_str = type_node.text.decode('utf8')
                        if type_str in ['sync.Mutex', 'sync.RWMutex', 'sync.WaitGroup', 'sync.Cond','sync.Once']:
                            results.append({
                                'name': name_node.text.decode('utf8'),
                                'scope': current_scope,
                                'type': type_str
                            })

        # 5. Catching struct fields via usage (e.g., s.mu.Lock(), wg.Add(1))
        # Since we cannot easily trace types back to struct definitions without semantic analysis,
        # we infer the primitive type based on the method called on it.
        elif node.type == 'call_expression':
            func_node = node.child_by_field_name('function')
            if func_node and func_node.type == 'selector_expression':
                operand = func_node.child_by_field_name('operand')
                field = func_node.child_by_field_name('field')
                
                if operand and field:
                    operand_name = operand.text.decode('utf8')
                    method_name = field.text.decode('utf8')
                    

                    if operand and operand.text == b'atomic':
                        # The first argument is the identifier being modified
                        args = node.child_by_field_name('arguments')
                        if args and len(args.children) > 1:
                            # child 1 is typically the first argument after '('
                            target = args.children[1].text.decode('utf8').lstrip('&')
                            results.append({
                                'name': target,
                                'scope': current_scope,
                                'type': f"atomic.{field.text.decode('utf8')}"
                            })
                    if method_name in ['Lock', 'Unlock']:
                        results.append({
                            'name': operand_name,
                            'scope': current_scope,
                            'type': 'sync.Mutex (inferred)'
                        })
                    elif method_name in ['Add', 'Wait', 'Done']:
                        results.append({
                            'name': operand_name,
                            'scope': current_scope,
                            'type': 'sync.WaitGroup (inferred)'
                        })
                    elif method_name in ['Signal', 'Broadcast']:
                        results.append({
                            'name': operand_name,
                            'scope': current_scope,
                            'type': 'sync.Cond (inferred)'
                        })
                    elif method_name in ['Do']:
                        results.append({
                            'name': operand_name,
                            'scope': current_scope,
                            'type': 'sync.Once (inferred)'
                        })

        # Recursively walk the AST
        for child in node.children:
            traverse(child, current_scope)

    traverse(root_node)
    
    # Deduplicate results (since usages like m.Lock() might appear multiple times in one scope)
    unique_results = [dict(t) for t in {tuple(d.items()) for d in results}]
    primitives_str = "\n".join([f"- {p['name']} ({p['type']}) in scope: {p['scope']}" for p in unique_results])
    return primitives_str


if __name__ == "__main__":        
    extracted_primitives = parse_go_concurrency(code=None,file_path="./benchmarks/goker/blocking/moby/27782/moby27782_test.go")
    print(extracted_primitives)