"""
Parse the exploration input from agent
"""
from typing import Optional
import re
import numpy as np

from ragen.env.spatial.Base.room import Room
from ragen.env.spatial.Base.action import ActionSequence
from ragen.env.spatial.Base.object import Object, Agent


def parse_action(action_str: str) -> Optional[ActionSequence]:
    """Parse action string into ActionSequence"""
    action_sequence = ActionSequence.parse(action_str)
    # Note: validation would be done by ExplorationManager during execution
    return action_sequence


if __name__ == "__main__":
    # Create test objects and room
    objects = [
        Object(name="table", pos=np.array([1, 1]), ori=np.array([1, 0])),
        Object(name="chair", pos=np.array([2, 2]), ori=np.array([0, 1])),
        Object(name="bookshelf", pos=np.array([0, 3]), ori=np.array([-1, 0])),
    ]
    agent = Agent()
    agent.name = "agent"
    agent.pos = np.array([0, 0])
    agent.ori = np.array([0, 1])
    
    test_room = Room(objects=objects, agent=agent, name="test_room")
    
    print("=== Testing parse_action function ===\n")
    
    # Test cases with expected results
    test_cases = [
        # Valid action strings
        ("Query(table)", True, "Simple query action"),
        ("Term()", True, "Termination action"),
        ("Move(chair); Query(table)", True, "Move then query"),
        ("Move(table), Rotate(90); Query(chair)", True, "Move, rotate, then query"),
        ("Return(); Query(bookshelf)", True, "Return then query"),
        ("Rotate(180); Query(table)", True, "Rotate the n query"),
        ("Move(chair), Move(table), Return(); Query(bookshelf)", True, "Multiple moves, return, then query"),
        
        # Invalid action strings
        ("Move(table), Query(chair)", False, "Motion and final action should be separated by semicolon"),
        ("Move(table); Move(chair)", False, "Move action as final action"),
        ("Query(table); Query(chair)", False, "Multiple final actions"),
        ("Term(); Query(table)", False, "Term should not have motion actions before it"),
        ("Move(table); Term(); Query(chair)", False, "Multiple semicolons not allowed"),
        ("Invalid(action)", False, "Invalid action format"),
        ("", False, "Empty string"),
        ("Move(table)", False, "Motion action without final action"),
        ("Rotate(90)", False, "Rotation action without final action"),
    ]
    
    # Run test cases
    passed = 0
    total = len(test_cases)
    
    for i, (action_str, expected_valid, description) in enumerate(test_cases, 1):
        result = parse_action(action_str)
        is_valid = result is not None
        
        status = "PASS" if is_valid == expected_valid else "FAIL"
        if status == "PASS":
            passed += 1
            
        print(f"Test {i:2d}: {status} - {description}")
        print(f"         Input: '{action_str}'")
        print(f"         Expected: {'Valid' if expected_valid else 'Invalid'}")
        print(f"         Got: {'Valid' if is_valid else 'Invalid'}")
        if is_valid:
            print(f"         Result: {result}")
        print()
    
    print(f"=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    # Additional detailed testing for debugging
    print("\n=== Detailed Action Parsing Examples ===")
    example_actions = [
        "Query(table)",
        "Move(chair); Query(table)", 
        "Move(table), Rotate(90); Query(chair)"
    ]
    
    for action_str in example_actions:
        print(f"\nParsing: '{action_str}'")
        parsed = ActionSequence.parse(action_str)
        if parsed:
            print(f"  Parsed successfully: {parsed}")
            print("  Motion actions:")
            for action in parsed.motion_actions:
                print(f"    - Type: {type(action).__name__}")
                print(f"    - Parameters: {action.parameters}")
            print("  Final action:")
            print(f"    - Type: {type(parsed.final_action).__name__}")
            print(f"    - Parameters: {parsed.final_action.parameters}")
            # Note: validation would be done by ExplorationManager
            print(f"  Validation: Skipped (would be done by ExplorationManager)")
        else:
            print("  Failed to parse")