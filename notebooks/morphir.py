import subprocess
import json

node_home='/root/.nvm/versions/node/v16.14.0'
lcr_directory='/tmp/lcr'




def __morphir_run(args):
  process = subprocess.run(
    args=args,
    cwd=lcr_directory, 
    env={'PATH': f'{node_home}/bin'},
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE,
    shell=False
  )
  return process

def morphir_compile():
  return __morphir_run([
      'morphir-elm',
      'make'
  ])

def __morphir_prepare_tests(test_input, test_output):
  test_payload = [[[[['regulation']],
   [['u', 's'], ['l', 'c', 'r'], ['inflows'], ['assets']],
   ['rule', '1', 'section', '20', 'c', '1']],
  [{}]]]
  test_payload[0][-1][0]['expectedOutput'] = test_output
  test_payload[0][-1][0]['inputs'] = [test_input]
  test_payload[0][-1][0]['description'] = 'GenAI test'
  with open(f'{lcr_directory}/morphir-tests.json', 'w') as f:
    f.write(json.dumps(test_payload))

def morphir_test(test_input, test_output):
  __morphir_prepare_tests(test_input, test_output)
  return __morphir_run([
      'morphir-elm',
      'test'
  ])


def morphir_test_distance(test_input, test_output):
  morphir_test_results = morphir_test(test_input, test_output)
  if morphir_test_results.returncode == 0:
    return 0
  else:   
    stderr = morphir_test_results.stdout.decode('UTF-8').split('\n')
    for s in stderr:
      if 'Actual Output' in s:
        # for out test, we expect output to be always integer
        # not sure that would be the case for each function
        return test_output - int(s.split(' ')[-1])
    raise('Could not find output matching distance')
  
def morphir_get_test_result(test_input):
  morphir_test_results = morphir_test(test_input, 0)
  console_output = morphir_test_results.stdout.decode('UTF-8').split('\n')
  try:
    for s in console_output:
      if 'Actual Output' in s:
        return int(s.split(' ')[-1])
  except:
    pass
  return None
