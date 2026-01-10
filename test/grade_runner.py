import sys
import os
import subprocess
import unittest
import site

# Step 1: Install requirements using the correct Python executable
# ✅ Step 2: Make sure user-installed packages are visible
sys.path.append(site.getusersitepackages())

# Step 3: Add the test directory to the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

# -----------------------------
# Step 3: Run a Specific Unit Test
# -----------------------------
def run_test(test_name):
    from unit_test import (
        TestFaissHNSW,
        TestFaissLSH,
        TestFlaskAppAdd
    )

    test_classes = [
        TestFaissHNSW,
        TestFaissLSH,
        TestFlaskAppAdd    ]

    suite = unittest.TestSuite()
    found = False
    for test_class in test_classes:
        if hasattr(test_class, test_name):
            suite.addTest(test_class(test_name))
            found = True
    if not found:
        print(f"FAIL (test '{test_name}' not found in any class)")
        exit(1)

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    print("PASS" if result.wasSuccessful() else "FAIL")
    exit(0 if result.wasSuccessful() else 1)
    

def run_test_2(test_name):
    from unit_test_2 import (
        TestFaissSearch,
        TestFlaskAppIdentify
    )

    test_classes = [
        TestFaissSearch,
        TestFlaskAppIdentify
    ]

    suite = unittest.TestSuite()
    found = False
    for test_class in test_classes:
        if hasattr(test_class, test_name):
            suite.addTest(test_class(test_name))
            found = True
    if not found:
        print(f"FAIL (test '{test_name}' not found in any class)")
        exit(1)

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    print("PASS" if result.wasSuccessful() else "FAIL")
    exit(0 if result.wasSuccessful() else 1)
    
# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grade_runner.py <test_method_name>")
        exit(1)

    test_method = sys.argv[1]
    type_test = int(sys.argv[2])

    try:

        if type_test == 1:
            run_test(test_method)
        else:
            run_test_2(test_method)
            
    except Exception as e:
        print(f"FAIL (exception: {e})")
        exit(1)
