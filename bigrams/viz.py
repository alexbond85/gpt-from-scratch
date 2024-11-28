from tabulate import tabulate


def print_matrix_section(matrix, tokenizer, max_size=7):
    chars = [tokenizer.decode_char(i) for i in range(min(max_size, tokenizer.vocab_size))]
    rows = []
    for i, row_char in enumerate(chars):
        probs = matrix.probabilities[i][:max_size]
        rows.append([row_char] + [f'{p:.2f}' for p in probs])

    print(tabulate(rows, headers=[' '] + chars, tablefmt='fancy_grid'))