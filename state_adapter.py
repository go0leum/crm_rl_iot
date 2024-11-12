from simple_observation_space import SimpleObservationSpace as Observation
class StateAdapter():
    def __init__(self):
        self.rides = [None, "equipment_1", "equipment_2", "equipment_3"]
        self.inventory_items = ["resource_1", "resource_2", "resource_3", "material_1", "material_2", "material_3"]
    def convert(self, agent_inventory, agent_ride, agent_payload, agent_location):
        state = Observation()
        for idx, item_name in enumerate(self.inventory_items):
            count = agent_inventory[item_name]
            state.agent_inventory[idx] = count

        for idx, val in enumerate(self.rides):
            if agent_ride == val:
                state.ride = idx

        state.agent_payload = agent_payload
        state.agent_location = agent_location.tolist()
        return state
