import grpc

from generated.griffig_pb2 import GraspRequest
from generated.griffig_pb2_grpc import GriffigStub


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = GriffigStub(channel)
        response = stub.CalculateGrasp(GraspRequest(method='max'))
    print(response)


if __name__ == '__main__':
    run()