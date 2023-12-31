"""Standard Heap implementation.
"""

import opfython.utils.constants as c
import opfython.utils.exception as e


class Heap:
    """A standard implementation of a Heap structure.

    """

    def __init__(self, size=1, policy='min'):
        """Initialization method.

        Args:
            size (int): Maximum size of the heap.
            policy (str): Heap's policy (`min` or `max`).

        """

        # Maximum size of the heap
        self.size = size

        # Policy that rules the heap
        self.policy = policy

        # List of node's costs
        self.cost = [c.FLOAT_MAX for i in range(size)]

        # List of node's colors
        self.color = [c.WHITE for i in range(size)]

        # List of node's values
        self.p = [-1 for i in range(size)]

        # List of node's positioning markers
        self.pos = [-1 for i in range(size)]

        # Last element identifier
        self.last = -1

    @property
    def size(self):
        """int: Maximum size of the heap.

        """

        return self._size

    @size.setter
    def size(self, size):
        if not isinstance(size, int):
            raise e.TypeError('`size` should be an integer')
        if size < 1:
            raise e.ValueError('`size` should be > 0')

        self._size = size

    @property
    def policy(self):
        """str: Policy that rules the heap.

        """

        return self._policy

    @policy.setter
    def policy(self, policy):
        if policy not in ['min', 'max']:
            raise e.ValueError('`policy` should be `min` or `max`')

        self._policy = policy

    @property
    def cost(self):
        """list: List of nodes' costs.

        """

        return self._cost

    @cost.setter
    def cost(self, cost):
        if not isinstance(cost, list):
            raise e.TypeError('`cost` should be a list')

        self._cost = cost

    @property
    def color(self):
        """list: List of nodes' colors.

        """

        return self._color

    @color.setter
    def color(self, color):
        if not isinstance(color, list):
            raise e.TypeError('`color` should be a list')

        self._color = color

    @property
    def p(self):
        """list: List of nodes' values.

        """

        return self._p

    @p.setter
    def p(self, p):
        if not isinstance(p, list):
            raise e.TypeError('`p` should be a list')

        self._p = p

    @property
    def pos(self):
        """list: List of nodes' positioning markers.

        """

        return self._pos

    @pos.setter
    def pos(self, pos):
        if not isinstance(pos, list):
            raise e.TypeError('`pos` should be a list')

        self._pos = pos

    @property
    def last(self):
        """int: Last element identifier.

        """

        return self._last

    @last.setter
    def last(self, last):
        if not isinstance(last, int):
            raise e.TypeError('`last` should be an integer')
        if last < -1:
            raise e.ValueError('`last` should be > -1')

        self._last = last

    def is_full(self):
        """Checks if the heap is full.

        Returns:
            A boolean indicating whether the heap is full.

        """

        if self.last == (self.size - 1):
            return True

        return False

    def is_empty(self):
        """Checks if the heap is empty.

        Returns:
            A boolean indicating whether the heap is empty.

        """

        if self.last == -1:
            return True

        return False

    def dad(self, i):
        """Gathers the position of the node's dad.

        Args:
            i (int): Node's position.

        Returns:
            The position of node's dad.

        """

        return int(((i - 1) / 2))

    def left_son(self, i):
        """Gathers the position of the node's left son.

        Args:
            i (int): Node's position.

        Returns:
            The position of node's left son

        """

        return int((2 * i + 1))

    def right_son(self, i):
        """Gathers the position of the node's right son.

        Args:
            i (int): Node's position.

        Returns:
            The position of node's right son.

        """

        return int((2 * i + 2))

    def go_up(self, i):
        """Goes up in the heap.

        Args:
            i (int): Position to be achieved.

        """

        # Gathers the dad's position
        j = self.dad(i)

        # Checks if policy is `min`
        if self.policy == 'min':
            # While the heap exists and the cost of post-node is bigger than current node
            while i > 0 and self.cost[self.p[j]] > self.cost[self.p[i]]:
                # Swap the positions
                self.p[j], self.p[i] = self.p[i], self.p[j]

                # Applies node's `i` and `j` values to the positioning list
                self.pos[self.p[i]] = i
                self.pos[self.p[j]] = j

                # Makes both indexes equal
                i = j

                # Gathers the new dad's position
                j = self.dad(i)

        # If policy is `max`
        else:
            # While the heap exists and the cost of post-node is smaller than current node
            while i > 0 and self.cost[self.p[j]] < self.cost[self.p[i]]:
                # Swap the positions
                self.p[j], self.p[i] = self.p[i], self.p[j]

                # Applies node's `i` and `j` values to the positioning list
                self.pos[self.p[i]] = i
                self.pos[self.p[j]] = j

                # Makes both indexes equal
                i = j

                # Gathers the new dad's position
                j = self.dad(i)

    def go_down(self, i):
        """Goes down in the heap.

        Args:
            i (int): Position to be achieved.

        """

        # Gathers the left and right son's positions
        left = self.left_son(i)
        right = self.right_son(i)

        # Equals the value of `j` and `i` counters
        j = i

        # Checks if policy is `min`
        if self.policy == 'min':
            # Checks if left node is not the last and its cost is smaller than previous
            if left <= self.last and self.cost[self.p[left]] < self.cost[self.p[i]]:
                # Apply `j` counter as the left node
                j = left

            # Checks if right node is not the last and its cost is smaller than previous
            if right <= self.last and self.cost[self.p[right]] < self.cost[self.p[j]]:
                # Apply `j` counter as the right node
                j = right

        # If policy is `max`
        else:
            # Checks if left node is not the last and its cost is bigger than previous
            if left <= self.last and self.cost[self.p[left]] > self.cost[self.p[i]]:
                # Apply `j` counter as the left node
                j = left

            # Checks if right node is not the last and its cost is bigger than previous
            if right <= self.last and self.cost[self.p[right]] > self.cost[self.p[j]]:
                # Apply `j` counter as the right node
                j = right

        # Checks if `j` is not equal to `i`
        if j != i:
            # Swap node's position
            self.p[j], self.p[i] = self.p[i], self.p[j]

            # Marks the new position in `i` and `j`
            self.pos[self.p[i]] = i
            self.pos[self.p[j]] = j

            # Goes down in the heap
            self.go_down(j)

    def insert(self, p):
        """Inserts a new node into the heap.

        Args:
            p (int): Node's value to be inserted.

        Returns:
            Boolean indicating whether insertion was performed correctly.

        """

        if not self.is_full():
            # Increases the last node's counter
            self.last += 1

            # Adds the new node to the heap, mark it as gray and mark its positioning
            self.p[self.last] = p
            self.color[p] = c.GRAY
            self.pos[p] = self.last

            self.go_up(self.last)

            return True

        return False

    def remove(self):
        """Removes a node from the heap.

        Returns:
            The removed node value.

        """

        if not self.is_empty():
            # Gathers the node's value
            p = self.p[0]

            # Marks it as not positioned and black-color
            self.pos[p] = -1
            self.color[p] = c.BLACK

            # Gathers the new position of first node
            self.p[0] = self.p[self.last]

            # Marks it as positioned and remove its value
            self.pos[self.p[0]] = 0
            self.p[self.last] = -1

            # Decreases the last counter
            self.last -= 1

            self.go_down(0)

            return p

        return False

    def update(self, p, cost):
        """Updates a node with a new value.

        Args:
            p (int): Node's position.
            cost (float): Node's cost.

        """

        # Applies the new cost
        self.cost[p] = cost

        # Checks if node has been removed or not
        if self.color[p] == c.BLACK:
            pass

        # Checks if node has not been inserted yet
        if self.color[p] == c.WHITE:
            self.insert(p)

        else:
            self.go_up(self.pos[p])
