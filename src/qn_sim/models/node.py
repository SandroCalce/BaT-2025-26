from numpy.random import default_rng
from networkx import Graph, shortest_path

from .memory import Memory
from ..protocols.generation import AdaptiveGenerationProtocol, PowerLawGenerationProtocol, UniformGenerationProtocol

class Node:
    """Class of network nodes.

    Hold quantum memories, information of total network topology (global variable) and nearest neighbor entanglement.
    Carry continuous entanglement generation, adaptive update, and path finding protocols.

    Attributes:
        label (int): integer to label the node, corresponding to the indices of traffic matrix and requests
        other_nodes (List[Node]): list of other node objects
        memo_size (int): number of quantum memories in the node, assuming memories are of the same type
        memories (List[Memory]): local memory objects.
        lifetime (int): quantum memory lifetime in unit of simulation time step, represents time to store entanglement
        entanglement_link_nums (Dict[int, int]): keeps track of numbers of entanglement links with direct neighbors (for path finding alg.)
        _next_avail_memory (int): index (in self.memories) of next memory that may be reserved.
        left_neighbors_to_connect (List[List]): list of left neighbors' indices in route for entanglement connection
        right_neighbors_to_connect (List[List]): list of right neighbors' indices in route for entanglement connection
        generation_protocol (GenerationProtocol): entanglement generation protocol attached to the node
    """

    def __init__(self, label, memo_size, lifetime, gen_success_prob, swap_success_prob, network, seed=0):
        """Constructor of a node instance.

        Args:
            label (int): integer to label the node, corresponding to the indices of traffic matrix and requests
            memo_size (int): number of quantum memories in the node, assuming memories are of the same type
            lifetime (int): quantum memory lifetime in unit of simulation time step, represents time to store entanglement
            gen_success_prob (float): success probability of entanglement generation between 0 and 1
            swap_success_prob (float): success probability of entanglement swapping between 0 and 1
            seed (int): seed for random number generators (default 0)
        """

        self.label = label
        self.other_nodes = []
        self.memo_size = memo_size
        self.memories = []
        self.entanglement_link_nums = {}
        self.left_neighbors_to_connect = []
        self.right_neighbors_to_connect = []
        self.sc_left_neighbors_to_connect = []
        self.sc_right_neighbors_to_connect = []

        self.generation_protocol = None

        self._next_avail_memory = 0

        # create memories
        for i in range(memo_size):
            memory = Memory("Node" + str(self.label) + "[%d]" % i, lifetime)
            memory.set_owner(self)
            self.memories.append(memory)

        # create rng and store params
        self.rng = default_rng(seed)
        self.gen_success_prob = gen_success_prob
        self.swap_success_prob = swap_success_prob

        self.network = network
        self.graph = Graph(network)

    def set_other_nodes(self, nodes):
        self.other_nodes = nodes
        self.entanglement_link_nums = {n.label: 0 for n in nodes}

    def set_generation_protocol(self, protocol_type, adapt_param):
        if protocol_type == "adaptive":
            neighbors = [j for j, element in enumerate(self.network[self.label]) if element != 0]
            self.generation_protocol = AdaptiveGenerationProtocol(self, adapt_param, neighbors)
        elif protocol_type == "powerlaw":
            self.generation_protocol = PowerLawGenerationProtocol(self, self.network)
        elif protocol_type == "uniform":
            self.generation_protocol = UniformGenerationProtocol(self, self.network)
        else:
            raise ValueError("Invalid generation type " + protocol_type)

    def memo_reserve(self, current_time):
        """Method for entanglement generation and swapping protocol to invoke to reserve quantum memories.

        Returns:
            Memory: memory object reserved (None if there are no free memories).
        """

        temp_idx = self._next_avail_memory
        while temp_idx < self.memo_size:
            mem = self.memories[temp_idx]

            is_fresh = (mem.last_update_time < current_time)
            # CHECK: If it's not reserved AND not a locked fixture
            if not mem.reserved and not mem.sc_fixed and is_fresh:
                mem.reserved = True
                self._next_avail_memory = temp_idx + 1
                return mem
            temp_idx += 1
        return None

    def memo_free(self, memory):
        """Method to free an occupied memory.

        Args:
            memory (Memory): memory object to free.
        """

        idx = self.memories.index(memory)
        memory.free()
        if idx < self._next_avail_memory:
            self._next_avail_memory = idx

    def memo_expire(self, memory):
        # avoid infinite loop
        if memory is None:
            return 
        if not memory.reserved:
            return

        other_node = memory.entangled_memory["node"]
        other_memory = memory.entangled_memory["memo"]

        self.entanglement_link_nums[other_node.label] -= 1
        memory.expire()
        self.memo_free(memory)

        other_node.memo_expire(other_memory)

    def create_random_link(self, time):
        label = self.generation_protocol.choose_link()
        other_node = next((n for n in self.other_nodes if n.label == label), None)
        self.create_link(time, other_node)

    def create_link(self, time, other_node):
        """Method to create an entanglement link with another node.

        If creation fails, will return False.

        Args:
            time (int): time of link creation (from main simulation loop).
            other_node (Node): node to generate entanglement with.

        Returns:
            bool: if creation succeeded (True) or failed (False).
        """

        # check if entanglement succeeds
        distance = len(shortest_path(self.graph, self.label, other_node.label)) - 1
        success_prob = (self.gen_success_prob ** distance) * (self.swap_success_prob ** (distance - 1))
        if self.rng.random() > success_prob:
            return False

        # reserve a local memory and a memory on the other node to entangle
        # Note: it is possible that when generating entanglement on demand, no memory is available for reservation
        local_memo = self.memo_reserve(time)
        if local_memo is None:
            return False
        other_memo = other_node.memo_reserve(time)
        if other_memo is None:
            self.memo_free(local_memo)
            return False

        # entangle the two nodes
        local_memo.entangle(other_memo, time)

        # record entanglement
        self.entanglement_link_nums[other_node.label] += 1
        # the other node should also update its entanglement link information
        other_node.entanglement_link_nums[self.label] += 1

        return True

    def create_link_with_priority(self, time, other_node):
        """Method to create an entanglement link with another node.

        If there are no memories available on local or destination node, will randomly pick one to overwrite.
        Entanglement may still fail due to random nature.

        Args:
            time (int): time of link creation (from main simulation loop).
            other_node (Node): node to generate entanglement with.
        """

        # --- 1. Reserve Local Memory ---
        local_memo = self.memo_reserve(time)
        if local_memo is None:
            # Try to find a non-fixed memory to expire
            valid_memo = [id for id, mem in enumerate(self.memories) if not mem.sc_fixed]
            if len(valid_memo) > 0:
                memo_id = self.rng.choice(valid_memo)
                self.memo_expire(self.memories[memo_id])
                local_memo = self.memo_reserve(time)

        # [FIX] If still None, we cannot proceed.
        if local_memo is None:
            return False

        # --- 2. Reserve Other Node Memory ---
        other_memo = other_node.memo_reserve(time)
        if other_memo is None:
            valid_memo = [id for id, mem in enumerate(other_node.memories) if not mem.sc_fixed]
            if len(valid_memo) > 0:
                memo_id = other_node.rng.choice(valid_memo)
                other_node.memo_expire(other_node.memories[memo_id])
                other_memo = other_node.memo_reserve(time)

        # [FIX] If other_memo is still None, cleanup local and abort.
        if other_memo is None:
            self.memo_free(local_memo)
            return False

        # --- 3. Generation Attempt ---
        if self.rng.random() > self.gen_success_prob:
            # [SAFE] We know both are valid memories now, so memo_free won't crash
            self.memo_free(local_memo)
            other_node.memo_free(other_memo)
            return False

        # --- 4. Success ---
        local_memo.entangle(other_memo, time)

        # record entanglement
        self.entanglement_link_nums[other_node.label] += 1
        other_node.entanglement_link_nums[self.label] += 1

        return True

    def swap(self, memory1, memory2, sc_start, sc_end, current_time):
        """Method to do entanglement swapping.

        Will reset the two involved memories' entanglement state.
        Will modify entanglement state of original entangled parties of memory1 and memory2.
        Does not modify start_time, and expiration of entanglement is determined by the first memory expiration

        Return the result of swapping (successful or not).
        """

        assert memory1 in self.memories and memory2 in self.memories

        if not memory1.reserved or not memory2.reserved:
            return False

        # Additional safety check: make sure memories are actually entangled
        if memory1.entangled_memory["node"] is None or memory2.entangled_memory["node"] is None:
            return False

        if memory1.last_update_time >= current_time or memory2.last_update_time >= current_time:
            return False

        if sc_start is not None and sc_end is not None:

            # Get labels for easier comparison
            n1_label = memory1.entangled_memory['node'].label
            n2_label = memory2.entangled_memory['node'].label
            start_label = sc_start.label
            end_label = sc_end.label
            my_label = self.label

            # Check connectivity
            # Does one memory connect to Start?
            has_link_to_start = (n1_label == start_label or n2_label == start_label)
            # Does one memory connect to End?
            has_link_to_end = (n1_label == end_label or n2_label == end_label)

            # --- CASE 1: ESTABLISHING THE SHORTCUT (_swap_sc) ---
            # Condition: I have one link to Start AND one link to End.
            # (Note: This usually happens at a middle node, but could happen anywhere bridging the two)
            if has_link_to_start and has_link_to_end:
                # This swap connects sc_start to sc_end directly.
                # It creates the "Fixed" infrastructure.
                return self._swap_sc(memory1, memory2)

            # --- CASE 2: USING THE SHORTCUT (_swap_sc_one) ---
            # Condition: I AM the Start or I AM the End node.
            i_am_start = (my_label == start_label)
            i_am_end = (my_label == end_label)

            if i_am_start or i_am_end:
                # If I am Start, I look for a link to End.
                # If I am End, I look for a link to Start.
                target_label = end_label if i_am_start else start_label
                target_node_obj = sc_end if i_am_start else sc_start

                # Check if one of the memories is the Fixed Shortcut Link
                mem1_is_sc = (n1_label == target_label)
                mem2_is_sc = (n2_label == target_label)

                if mem1_is_sc or mem2_is_sc:
                    # Additional check: is the shortcut memory actually fixed?
                    sc_memory = memory1 if mem1_is_sc else memory2

                    # Only use _swap_sc_one if the shortcut is still fixed
                    if sc_memory.sc_fixed:
                        # We are at an endpoint, swapping the fixed link with a dynamic neighbor.
                        # This executes the "Copy and Transfer" logic.
                        return self._swap_sc_one(memory1, memory2, target_node_obj)
                    # Otherwise fall through to regular swap

        memo1 = memory1.entangled_memory["memo"]
        memo2 = memory2.entangled_memory["memo"]
        node1 = memory1.entangled_memory["node"]
        node2 = memory2.entangled_memory["node"]

        if self.rng.random() < self.swap_success_prob:
            # reset local entanglement
            memory1.expire()
            memory2.expire()
            self.memo_free(memory1)
            self.memo_free(memory2)
            self.entanglement_link_nums[node1.label] -= 1
            self.entanglement_link_nums[node2.label] -= 1

            # entanglement connection, maintain same expiration time
            memo1.entangled_memory["node"] = node2
            memo2.entangled_memory["node"] = node1
            memo1.entangled_memory["memo"] = memo2
            memo2.entangled_memory["memo"] = memo1

            # update entanglement count
            node1.entanglement_link_nums[self.label] -= 1
            node2.entanglement_link_nums[self.label] -= 1
            node1.entanglement_link_nums[node2.label] += 1
            node2.entanglement_link_nums[node1.label] += 1

            return True

        else:
            # if unsuccessful, all involved memories entanglement reset
            node1.memo_expire(memo1)
            node2.memo_expire(memo2)
            return False

    def _swap_sc(self, memory1, memory2):
        memo1 = memory1.entangled_memory["memo"]
        memo2 = memory2.entangled_memory["memo"]
        node1 = memory1.entangled_memory["node"]
        node2 = memory2.entangled_memory["node"]
        if self.rng.random() < self.swap_success_prob:
            # reset local entanglement
            memory1.expire()
            memory2.expire()
            self.memo_free(memory1)
            self.memo_free(memory2)
            self.entanglement_link_nums[node1.label] -= 1
            self.entanglement_link_nums[node2.label] -= 1

            memo1.sc_fixed = True
            memo2.sc_fixed = True
            # entanglement connection, maintain same expiration time
            memo1.entangled_memory["node"] = node2
            memo2.entangled_memory["node"] = node1
            memo1.entangled_memory["memo"] = memo2
            memo2.entangled_memory["memo"] = memo1

            # update entanglement count
            node1.entanglement_link_nums[self.label] -= 1
            node2.entanglement_link_nums[self.label] -= 1
            node1.entanglement_link_nums[node2.label] += 1
            node2.entanglement_link_nums[node1.label] += 1

            return True

        else:
            # if unsuccessful, all involved memories entanglement reset
            node1.memo_expire(memo1)
            node2.memo_expire(memo2)
            return False

    def _swap_sc_one(self, memory1, memory2, sc_node):
        """
        Swaps a shortcut memory with a dynamic memory, RELEASING the shortcut link.

        Effect:
        1. Creates a NEW direct link between (SC_Node <-> Path_Node).
        2. RELEASES both local and remote shortcut memories (unfixes them).
        3. Frees the local dynamic memory used for the path.
        """

        # 1. Identify which memory is the Shortcut (Fixed) and which is the Path (Dynamic)
        if memory1.entangled_memory["node"].label == sc_node.label:
            mem_sc_local = memory1
            mem_path_local = memory2
        else:
            mem_sc_local = memory2
            mem_path_local = memory1

        # Get the remote nodes and memories
        remote_sc_node = mem_sc_local.entangled_memory["node"]  # e.g., Node E
        remote_sc_memo = mem_sc_local.entangled_memory["memo"]  # The memory on Node E
        remote_path_node = mem_path_local.entangled_memory["node"]  # e.g., Node C
        remote_path_memo = mem_path_local.entangled_memory["memo"]  # The memory on Node C

        # Check if we'd be creating a duplicate link
        if remote_sc_node.entanglement_link_nums.get(remote_path_node.label, 0) > 0:
            return False

        if self.rng.random() < self.swap_success_prob:
            # --- SUCCESSFUL SWAP ---

            # Step A: Update the remote shortcut memory to connect to path node
            # Keep this memory reserved and entangled
            expire_time = remote_path_memo.entangled_memory["expire_time"]

            remote_sc_memo.entangled_memory["node"] = remote_path_node
            remote_sc_memo.entangled_memory["memo"] = remote_path_memo
            remote_sc_memo.entangled_memory["expire_time"] = expire_time
            remote_sc_memo.sc_fixed = False  # No longer a fixed shortcut
            # IMPORTANT: Keep remote_sc_memo.reserved = True

            # Update the path node's memory to point to the SC node
            remote_path_memo.entangled_memory["node"] = remote_sc_node
            remote_path_memo.entangled_memory["memo"] = remote_sc_memo
            # remote_path_memo stays reserved as it was

            # Step B: Release the LOCAL shortcut memory
            mem_sc_local.sc_fixed = False
            mem_sc_local.expire()
            self.memo_free(mem_sc_local)

            # Step C: Release the LOCAL path memory
            mem_path_local.expire()
            self.memo_free(mem_path_local)

            # Step D: Update Topology/Stats
            # Self (Node A) lost connection to SC Node (Node E)
            self.entanglement_link_nums[remote_sc_node.label] -= 1
            remote_sc_node.entanglement_link_nums[self.label] -= 1

            # Self (Node A) lost connection to Path Node (Node C)
            self.entanglement_link_nums[remote_path_node.label] -= 1
            remote_path_node.entanglement_link_nums[self.label] -= 1

            # SC Node (Node E) gained connection to Path Node (Node C)
            remote_sc_node.entanglement_link_nums[remote_path_node.label] += 1
            remote_path_node.entanglement_link_nums[remote_sc_node.label] += 1

            return True

        else:
            # --- FAILURE ---
            # On failure, both memories are destroyed

            # Break the shortcut link
            remote_sc_memo.sc_fixed = False
            remote_sc_node.memo_expire(remote_sc_memo)

            mem_sc_local.sc_fixed = False
            mem_sc_local.expire()
            self.memo_free(mem_sc_local)

            # This will also update topology counters for A-E link
            self.entanglement_link_nums[remote_sc_node.label] -= 1
            remote_sc_node.entanglement_link_nums[self.label] -= 1

            # Break the path connection (this calls memo_expire which handles the other node)
            remote_path_node.memo_expire(remote_path_memo)

            return False
