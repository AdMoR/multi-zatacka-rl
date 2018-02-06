
import json
import math
import random

COLORS = ['#f00', '#0f0', '#00f', '#ff0', '#f0f', '#0ff']


class Snake(object):

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.speed = 2
        self.turn_speed = 0.05
        self.radius = 2
        self.direction = random.random() * 2 * math.pi
        self.old_x = self.x
        self.old_y = self.y

    def move(self, width, height):
        self.old_x = self.x
        self.old_y = self.y
        self.x += self.speed * math.cos(self.direction)
        self.y -= self.speed * math.sin(self.direction)

        if self.x > width:
            self.x = 0
        if self.y > height:
            self.y = 0
        if self.x < 0:
            self.x = width
        if self.y < 0:
            self.y = height

    def turn_right(self):
        self.direction -= self.turn_speed

    def turn_left(self):
        self.direction += self.turn_speed

    def update_grid(self, grid):
        x = self.old_x
        y = self.old_y
        for w in range(2 * self.radius):
            for h in range(2 * self.radius):
                xx = int(round(x - self.radius + w))
                yy = int(round(y - self.radius + h))
                grid.set(xx, yy, self.id)

    def collision(self, grid):
        sensors_x = [int(round(
            self.x + self.radius * math.cos(self.direction + math.pi * i / 6)))
            for i in range(-2, 3)]
        sensors_y = [int(round(
            self.y - self.radius * math.sin(self.direction + math.pi * i / 6)))
            for i in range(-2, 3)]

        if any(grid.get(x, y) for (x, y) in zip(sensors_x, sensors_y)):
            s = ''
            xx = (min(sensors_x) - 10, max(sensors_x) + 10)
            yy = (min(sensors_y) - 10, max(sensors_y) + 10)
            s += 'id: %d\n' % self.id
            s += 'xx: %s, yy: %s\n' % (xx, yy)
            s += 'x, y: %d, %d   old: %d, %d\n' % (self.x, self.y, self.old_x, self.old_y)
            #for y in range(*yy):
            #    s += ''.join(
            #        '%d%s' % (grid.get(x, y),
            #            '*' if x == int(self.x) and y == int(self.y)
            #            else 'X' if x == int(self.old_x) and y == int(self.old_y)
            #            else '!' if (x, y) in zip(sensors_x, sensors_y)
            #            else ' ')
            #        for x in range(*xx)
            #    )
            #    s += '\n'
            print(s)
            return True


class AbstractPlayer(object):

    def __init__(self, id):
        raise NotImplementedError

    def spawn(self, x, y):
        raise NotImplementedError

    def process(self, message, game_state):
        '''
        Function used by human players to pass actions
        '''
        raise NotImplementedError

    def update(self, grid):
        '''
        Prepare next move depending on action for humans or grid state for bots
        '''
        raise NotImplementedError

    def get_snake(self):
        raise NotImplementedError

