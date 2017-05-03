assignments = []

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    for unit in unitlist:
        unit_twins = [] #Create an empty list that will hold our naked twins
        for box in unit:
            #If the box has a value length of 2, check if it has a twin in the unit and add it to the twins list
            if len(values[box]) == 2:
                for twin in unit:
                    if twin != box and values[box] == values[twin]:
                        unit_twins.append([box, twin])
                        
        #Loop over naked twins and remove their values from the other boxes in the unit
        for twin in unit_twins:
            for digit in values[twin[0]]:
                for box in unit:
                    if box != twin[0] and box != twin[1]: #Only remove values from non naked twins.
                        values = assign_value(values, box, values[box].replace(digit,''))
    return values

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]

rows = 'ABCDEFGHI'
cols = '123456789'
    
#Setup basic naming conventions for boxes and units
boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
diag_down = [a[0]+a[1] for a in zip('ABCDEFGHI', '123456789')]
diag_up = [a[0]+a[1] for a in zip('ABCDEFGHI', '123456789'[::-1])]
unitlist = row_units + column_units + square_units + [diag_down] + [diag_up]
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

def grid_values(grid):
    
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    
    assert len(grid) == 81
    
    values = []
    for value in grid:
        if value == '.':
            values.append('123456789')
        else:
            values.append(value) 
    return dict(zip(boxes, values))

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):
    """
    Remove values from peers that are already solved
    Args:
        values(dict): The sudoku in dictionary form
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values = assign_value(values, peer, values[peer].replace(digit,''))
    return values

def only_choice(values):
    """
    Solve boxes that only have 1 possiable solution under the current contraints 
    Args:
        values(dict): The sudoku in dictionary form
    """
    digits = '123456789'
    for unit in unitlist:
        for digit in digits:
            digit_loc = [box for box in unit if digit in values[box]]
            if len(digit_loc) == 1:
                values = assign_value(values, digit_loc[0], digit)
    return values
        

def reduce_puzzle(values):
    """
    Loop over elimination, only choice and naked twin stratergies to reduce the puzzle
    Args:
        values(dict): The sudoku in dictionary form
    """
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        
        #Eliminate Strategy
        values = eliminate(values)

        #Only Choice Strategy
        values = only_choice(values)
        
        #Naked Twins Strategy
        values = naked_twins(values)
        
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    
    if values is False:
        return False ## Failed earlier
    
    if all(len(values[box]) == 1 for box in boxes): 
        return values ## Solved!
    
    # Choose one of the unfilled squares with the fewest possibilities
    length, box = min((len(values[box]), box )for box in boxes if len(values[box]) > 1)
    
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for value in values[box]:
        new_sudoku = values.copy()
        new_sudoku[box] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    
    #Solve the Sudoku
    values = search(grid_values(grid))
    return values

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
