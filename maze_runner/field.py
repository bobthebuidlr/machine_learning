import numpy as np


class Field:

    """
    Initialize the field as a square, with a certain horizontal X and vertical Y
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.field = np.zeros(shape=(self.width, self.height))

    """
    Draw a line of 1's across the given start and end (1 dimensional)
    """
    def draw_line(self, start, end):

        if start.x == end.x:
            direction = 'vertical'
            distance = np.abs(start.y - end.y) + 1

        elif start.y == end.y:
            direction = 'horizontal'
            distance = np.abs(start.x - end.x) + 1

        else:
            raise ValueError("Either the X or Y has to be the same dimension for the start and end")

        print(distance)
        print(direction)

        # TODO: Figure out the right calculations for drawring straight lines

        # if direction == 'horizontal':
        #     for i in range(distance):
        #         self.field[start.y][i + start.x] = 1
        #
        # elif direction == 'vertical':
        #     for i in range(distance):
        #         self.field[start.y + i][start.y] = 1


class Coordinate:

    def __init__(self, x, y):
        self.x = x
        self.y = y


field = Field(100, 100)
# field.draw_line(start=Coordinate(2, 2), end=Coordinate(99, 2))
field.draw_line(start=Coordinate(99, 2), end=Coordinate(2, 99))
# field.draw_line(start=Coordinate(99, 99), end=Coordinate(99, 2))
field.draw_line(start=Coordinate(99, 2), end=Coordinate(2, 2))
print(field.field)