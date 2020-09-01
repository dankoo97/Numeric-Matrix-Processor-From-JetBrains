
class Matrix:

    def set_size(self):
        self.size['y'], self.size['x'] = [int(_) for _ in input().split()]

    def set_matrix(self):
        self.matrix = []

        for row in range(self.size['y']):
            self.matrix.append([float(_) for _ in input().split()])

    def has_float(self):
        for row in range(self.size['y']):
            for column in range(self.size['x']):
                if not self.matrix[row][column].is_integer():
                    return True
        return False

    def convert_to_int(self):
        for row in range(self.size['y']):
            for column in range(self.size['x']):
                self.matrix[row][column] = int(self.matrix[row][column])
        return self

    def check_and_change_to_int(self):
        if self.has_float():
            return self
        return self.convert_to_int()

    def is_vector(self):
        return self.size['x'] == 1 or self.size['y'] == 1

    def is_scalar(self):
        return self.size['x'] == 1 and self.size['y'] == 1

    def vector_multiply(self, other):
        val = 0

        for index in range(len(self.matrix)):
            val += self.matrix[index] * other.matrix[index]

        return val

    def get_row(self, row):
        new_matrix = Matrix({'x': self.size['x'], 'y': 1})
        new_matrix.matrix = self.matrix[row]

        return new_matrix

    def get_col(self, col):
        new_matrix = Matrix({'x': 1, 'y': self.size['y']})
        new_matrix.matrix = []

        for row in range(new_matrix.size['y']):
            new_matrix.matrix.append(self.matrix[row][col])

        return new_matrix

    def clear_matrix(self):
        self.size = {'x': 0, 'y': 0}
        self.matrix.clear()
        return self

    def is_empty(self):
        return 0 not in self.size.values()

    def transpose(self, line):
        transposed_matrix = Matrix(self.size)
        if line[0] == 'm':
            for row in range(transposed_matrix.size['y']):
                transposed_matrix.matrix.append([])
            for row in range(self.size['y']):
                for col in range(self.size['x']):
                    transposed_matrix.matrix[col].append(self.matrix[row][col])

        elif line[0] == 's':
            for row in range(transposed_matrix.size['y']):
                transposed_matrix.matrix.append([])
            for row in range(self.size['y']):
                for col in range(self.size['x']):
                    transposed_matrix.matrix[col].append(self.matrix[-row - 1][-col - 1])

        elif line[0] == 'v':
            for row in self.matrix:
                transposed_matrix.matrix.append(row[::-1])

        elif line[0] == 'h':
            for row in self.matrix:
                transposed_matrix.matrix.insert(0, row)

        return transposed_matrix

    def __init__(self, size=None):
        self.matrix = []
        if size:
            self.size = size
        else:
            self.size = {'x': 0, 'y': 0}

    def __add__(self, other):
        if self.size != other.size:
            return Matrix()

        new_matrix = Matrix(self.size)

        for row in range(self.size['y']):
            new_matrix.matrix.append([])
            for column in range(self.size['x']):
                new_matrix.matrix[row].append(self.matrix[row][column] + other.matrix[row][column])

        return new_matrix

    def __mul__(self, other):

        new_matrix = Matrix()

        if isinstance(other, int) or isinstance(other, float) or other.is_scalar():
            new_matrix = Matrix(self.size)
            for row in range(self.size['y']):
                new_matrix.matrix.append([])
                for column in range(self.size['x']):
                    new_matrix.matrix[row].append(other * self.matrix[row][column])

        elif isinstance(other, Matrix):

            if self.size['x'] != other.size['y']:
                return 'The operation cannot be performed.'

            new_matrix = Matrix({'x': other.size['x'], 'y': self.size['y']})

            if self.is_vector() and other.is_vector():
                return self.vector_multiply(other)

            for row in range(new_matrix.size['y']):
                new_matrix.matrix.append([])

                for col in range(new_matrix.size['x']):
                    new_matrix.matrix[row].append(self.get_row(row) * other.get_col(col))

        return new_matrix

    def __str__(self):
        if 0 in self.size:
            return 'ERROR'
        matrix_print = ''
        for row in self.matrix:
            matrix_print += ' '.join(str(_) for _ in row) + '\n'
        return matrix_print

    def determinant(self):
        if self.size['x'] == 1:
            return self.matrix[0][0]
        if self.size['x'] == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        det = 0
        for i, val in enumerate(self.matrix[0]):
            cofactor_matrix = self.cofactor(0, i)
            det += val * cofactor_matrix.determinant() * (-1) ** i
            # print('det:', det)
            # print('val:', val)
            # print('cofactor:\n' + str(cofactor_matrix))

            cofactor_matrix.clear_matrix()
        return det

    def cofactor(self, i, j):
        cofactor_matrix = Matrix({'x': self.size['x'] - 1, 'y': self.size['y'] - 1})
        for row in self.matrix[:i] + self.matrix[i + 1:]:
            # print(row)
            cofactor_matrix.matrix.append(row[:j] + row[j + 1:])
        return cofactor_matrix

    def inverse(self):
        if self.determinant() == 0:
            return "This matrix doesn't have an inverse."
        return self.adjugate() * (1 / self.determinant())

    def adjugate(self):
        minors = Matrix(self.size)
        for j, row in enumerate(self.matrix):
            minors.matrix.append([])
            for i, col in enumerate(row):
                minors.matrix[j].append(self.cofactor(i, j).determinant() * (-1) ** (i + j))
        # print(minors)
        return minors


# test_matrix = Matrix({'x': 3, 'y': 3})
# test_matrix.matrix = [
#     [3, 0, 2],
#     [2, 0, -2],
#     [0, 1, 1],
# ]
#
# print(test_matrix.inverse())

menu = {
    'main_menu': '1. Add matrices\n2. Multiply matrix by a constant\n3. Multiply matrices\n4. Transpose matrix\n5. Calculate a determinant\n6. Inverse matrix\n0. Exit\nYour choice:',
    'set_size': 'Enter size of matrix:',
    'set_matrix': 'Enter matrix:',
    'constant': 'Enter constant:',
    'result': 'The result is:',
    'transpose': '1. Main diagonal\n2. Side diagonal\n3. Vertical line\n4. Horizontal line\nYour choice:',
}

transpose_options = ['main_diag', 'side_diag', 'vert', 'horiz']

state = 'main_menu'

matrix_1 = Matrix()
matrix_2 = Matrix()
operation = ''

while operation != 'exit':
    try:
        print(menu[state])
    except KeyError:
        print(menu[state[:-1]])
    if state == 'main_menu':
        action = int(input())
        operation = ('exit', '+', '*c', '*m', 't', 'd', 'i')[action]
        if operation == 't':
            print(menu['transpose'])
            operation = transpose_options[int(input()) - 1]
        state = 'set_size1'
    elif state[:-1] == 'set_size':
        if state[-1] == '1':
            matrix_1.set_size()
            state = 'set_matrix1'
        else:
            matrix_2.set_size()
            state = 'set_matrix2'
    elif state[:-1] == 'set_matrix':
        if state[-1] == '1':
            # print('matrix 1')
            matrix_1.set_matrix()
            if operation in ('+', '*m'):
                state = 'set_size2'
            elif operation == '*c':
                state = 'constant'
            elif operation[0] in ('m', 's', 'v', 'h', 'd', 'i'):
                state = 'result'
        else:
            # print('matrix 2')
            matrix_2.set_matrix()
            state = 'result'
    elif state == 'constant':
        matrix_2 = int(input())
        state = 'result'
    elif state == 'result':
        result_matrix = None
        if operation == '+':
            result_matrix = matrix_1 + matrix_2
        elif operation[0] == '*':
            result_matrix = matrix_1 * matrix_2
        elif operation[0] in ('m', 's', 'v', 'h'):
            result_matrix = matrix_1.transpose(operation)
        elif operation[0] == 'i':
            result_matrix = matrix_1.inverse()
        elif operation[0] == 'd':
            print(matrix_1.determinant())
            state = 'main_menu'
            continue
        result_matrix.check_and_change_to_int()
        print(result_matrix)
        result_matrix.clear_matrix()
        state = 'main_menu'
