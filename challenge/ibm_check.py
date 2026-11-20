"""IBM Quantum access check for teammates (hardware bonus). The token is never printed, logged or written to the repo.

    pip install -r requirements-hardware.txt
    python -m challenge.ibm_check --save     # asks for the API key (hidden input) and saves it to ~/.qiskit/qiskit-ibm.json
    python -m challenge.ibm_check            # verifies the saved account: backends, queues, remaining quota

Get the API key from https://quantum.cloud.ibm.com (channel 'ibm_quantum_platform'; the old 'ibm_quantum' channel is retired).
QISKIT_IBM_TOKEN in the environment is used instead of the prompt when set.
"""
import argparse
import getpass
import os


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    from qiskit_ibm_runtime import QiskitRuntimeService
    if args.save:
        token = os.environ.get('QISKIT_IBM_TOKEN') or getpass.getpass('IBM Quantum API key (hidden): ')
        QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token=token.strip(), overwrite=True, set_as_default=True)
        print('Saved to ~/.qiskit/qiskit-ibm.json (outside the repo; .qiskit/ is also in .gitignore).')
    service = QiskitRuntimeService()
    for b in service.backends(operational=True, simulator=False):
        print(f'{b.name:16s} {b.num_qubits:4d} qubits  pending jobs: {b.status().pending_jobs}')
    u = service.usage()
    print(f"Quota: {u.get('usage_remaining_seconds')} s left of {u.get('usage_limit_seconds')} s this period.")
    print('Access confirmed. The +5 bonus still needs a verified job: python -m challenge.qk_hardware submit ... then analyze.')


if __name__ == '__main__':
    main()
