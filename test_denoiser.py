import subprocess

OUTPUT_DIR = './outputs/tmp'

TEST_COMMANDS = {'demucs': ['train.py', 'dset=valentini_dummy', 'experiment=demucs_1', 'stft_loss=True',
                            'experiment.segment=2', 'experiment.stride=2','ddp=0', 'batch_size=16', 'experiment.scale_factor=2',
                            'eval_every=1', 'epochs=2', f'hydra.run.dir={OUTPUT_DIR}', 'device=cpu'],
                 }
REMOVE_OUTPUT_FILE_COMMAND = ['rm', '-r', OUTPUT_DIR]

successful_tests = []
failed_tests = []
outputs = []

def test_denoiser():
    for exp_name, command in TEST_COMMANDS.items():
        print('============================')
        print(f'running test: {exp_name}')
        try:
            output = subprocess.run(command, capture_output=True, text=True)
            output.check_returncode()
        except subprocess.CalledProcessError:
            print(f'{exp_name} failed!\n')
            failed_tests.append(exp_name)
        else:
            print(f'{exp_name} passed!\n')
            successful_tests.append(exp_name)
        finally:
            outputs.append(output)
            print(output.returncode)
            print(output.stdout)
            print(output.stderr)

        subprocess.run(REMOVE_OUTPUT_FILE_COMMAND)

    print(f'done running tests. {len(successful_tests)}/{len(TEST_COMMANDS)} tests passed.')
    print(f'successful tests: {successful_tests}')
    print(f'failed tests: {failed_tests}')

if __name__ == "__main__":
    test_denoiser()