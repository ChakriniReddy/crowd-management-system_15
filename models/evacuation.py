import numpy as np
import heapq

class EvacuationRouter:
    def __init__(self, logical_grid_size=10, camera_rows=2, camera_cols=2):
        self.grid_size = logical_grid_size
        self.cam_rows = camera_rows
        self.cam_cols = camera_cols
        
        # logical_grid is grid_size x grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # Start at bottom-right, exit at top-left.
        self.start = (self.grid_size - 1, self.grid_size - 1)
        self.exit = (0, 0)
        
    def _map_cam_to_logical(self, cam_r, cam_c):
        # Maps a camera grid region (0,0) to start_r, end_r, start_c, end_c on the logical grid
        section_h = self.grid_size // self.cam_rows
        section_w = self.grid_size // self.cam_cols
        
        start_r = cam_r * section_h
        end_r = start_r + section_h
        
        start_c = cam_c * section_w
        end_c = start_c + section_w
        
        return start_r, end_r, start_c, end_c

    def update_hazards(self, fire_grids, overflow_grids, fallen_grids):
        # Reset grid (0 = safe, 1 = hazard)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        all_hazards = list(set(fire_grids + overflow_grids + fallen_grids))
        
        for hazard in all_hazards:
            r, c = map(int, hazard.split('-'))
            sr, er, sc, ec = self._map_cam_to_logical(r, c)
            # Mark hazard with 1
            self.grid[sr:er, sc:ec] = 1
            
        return self._a_star()

    def _heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _a_star(self):
        # If exit is hazard, cannot escape
        if self.grid[self.exit] == 1:
            return []
            
        # If start is hazard, we still try to path out of it
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        close_set = set()
        came_from = {}
        gscore = {self.start: 0}
        fscore = {self.start: self._heuristic(self.start, self.exit)}
        
        oheap = []
        heapq.heappush(oheap, (fscore[self.start], self.start))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            
            if current == self.exit:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1] # Reverse

            close_set.add(current)
            
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                
                # Check bounds
                if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                    if self.grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                        
                    # Cost
                    tentative_g_score = gscore[current] + self._heuristic(current, neighbor)
                    
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                        continue
                        
                    if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + self._heuristic(neighbor, self.exit)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
                        
        return []
