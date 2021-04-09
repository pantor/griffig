from concurrent import futures

import grpc

from generated.helloworld_pb2 import HelloReply
from generated.helloworld_pb2_grpc import GreeterServicer, add_GreeterServicer_to_server


class Greeter(GreeterServicer):
    def SayHello(self, request, context):
        return HelloReply(message=f'Hello, {request.name}!')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
