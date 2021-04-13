from concurrent import futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import grpc
from generated.griffig_pb2 import GraspReply, Pose
from generated.griffig_pb2_grpc import GriffigServicer, add_GriffigServicer_to_server

from griffig import Griffig, RobotPose


class GriffiggRPC(GriffigServicer):
    def CalculateGrasp(self, request, context):
        grasp_pose = RobotPose(x=0.2, z=0.3, d=0.05)

        reply = GraspReply()
        reply.pose.x = grasp_pose.x
        reply.pose.y = grasp_pose.y
        reply.pose.z = grasp_pose.z
        reply.pose.a = grasp_pose.a
        reply.pose.b = grasp_pose.b
        reply.pose.c = grasp_pose.c
        reply.stroke = grasp_pose.d

        return reply


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_GriffigServicer_to_server(GriffiggRPC(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
