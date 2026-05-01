class Memory:
    """Simplified class of quantum memories to be stored in a node.

    Omitting details of memory efficiency, quantum state fidelity, photon wavelength, memory maximal frequency of reuse, etc.

    Attributes:
        name (str): name of a memory array instance
        owner (Node): node object which holds this memory.
        lifetime (int): quantum memory lifetime in unit of simulation time step, represents time to store quantum entanglement
        reserved (bool): indicates if the memory has been reserved by the owning node.
        sc_fixed (bool): indicates if the shortcuts uses this memory.
        entangled_memory (Dict[str, any]): records information on another memory sharing entanglement (if it exists).
    """

    def __init__(self, name, lifetime):
        """Constructor of memory instance.

        Args:
            name (str): name of memory instance
            lifetime (int): quantum memory lifetime in unit of simulation time step, represents time to store quantum entanglement
        """

        self.name = name
        self.owner = None
        self.lifetime = lifetime
        self.reserved = False  # Boolean representing if the memory has been reserved for use
        self.sc_fixed = False

        self.last_update_time = -1

        self.entangled_memory = {"node": None, "memo": None, "expire_time": None}

    def entangle(self, memory, time):
        self.entangled_memory = {"node": memory.owner, "memo": memory, "expire_time": time + self.lifetime}
        # the other memory should also update its entanglement information
        memory.entangled_memory = {"node": self.owner, "memo": self, "expire_time": time + memory.lifetime}
        self.last_update_time = time
        memory.last_update_time = time

    def set_owner(self, node):
        self.owner = node

    def reserve(self):
        if not self.reserved:
            self.reserved = True
        else:
            raise Exception("This memory has already been reserved")

    def free(self):
        if self.reserved:
            self.reserved = False
        else:
            raise Exception("This memory is not reserved")

    def expire(self):
        self.entangled_memory = {"node": None, "memo": None, "expire_time": None}
        self.sc_fixed = False
