/**
 *
 *  \file
 *  \brief      Main entry point for the socket server node.
 *  \author     Mike Purvis <mpurvis@clearpathrobotics.com>
 *  \copyright  Copyright (c) 2013, Clearpath Robotics, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Clearpath Robotics, Inc. nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CLEARPATH ROBOTICS, INC. BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Please send comments, questions, or patches to code@clearpathrobotics.com
 *
 */

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

#include <ros/ros.h>

#include <topic_tools/shape_shifter.h>
#include <std_msgs/Byte.h>
#include <sensor_msgs/PointCloud2.h>

#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

using namespace std;

#define PORT 8080

int socket_desc , client_sock;
bool bConnect;

void createTcpServer()
{
	struct sockaddr_in server;
	int opt = 1;	
	
	//Create socket
	socket_desc = socket(AF_INET , SOCK_STREAM , 0);
	if (socket_desc == -1)
	{
		printf("Could not create socket");
	}
	printf("Socket created");
	
	//Prepare the sockaddr_in structure
	server.sin_family = AF_INET;
	server.sin_addr.s_addr = INADDR_ANY;
	server.sin_port = htons( 8888 );
	
	//Bind
	if( bind(socket_desc,(struct sockaddr *)&server , sizeof(server)) < 0)
	{
		//print the error message
		printf("bind failed. Error");
		return;
	}
	printf("bind done");
	
	//Listen
	listen(socket_desc , 3);

}
void *acceptClient(void* arg)
{
	bConnect = false;
	int c;
	struct sockaddr_in client;
	//Accept and incoming connection
	printf("Waiting for incoming connections...");
	c = sizeof(struct sockaddr_in);
	
	//accept connection from an incoming client
	client_sock = accept(socket_desc, (struct sockaddr *)&client, (socklen_t*)&c);
	if (client_sock < 0)
	{
		printf("accept failed");
		return NULL;
	}
	printf("Connection accepted");
	bConnect = true;
	
}

void sendPointData(std::vector<uint8_t>& buffer)
{
	int send_size;
	uint32_t buffersize = buffer.size();
	char* client_message = new char[buffersize + 4];
	for(int i = 0; i < 4; i++)
	{
		*(client_message + i) = (buffersize >> (8*i)) & 0xFF;
	}
	std::copy(buffer.begin(), buffer.end(), client_message + 4);
	printf("client_message[0]=%02X,client_message[1]=%02X\n, client_message[2]=%02X,client_message[3]=%02X\n", client_message[0],client_message[1],client_message[2],client_message[3]);
	printf("length=%d, client_message[4]=%02X,client_message[final]=%02X\n", buffer.size()+4, client_message[4],client_message[buffer.size()+3]);
	if(!bConnect)
		return;

	send_size = write(client_sock , client_message , buffersize + 4);
	
	if(send_size == 0)
	{
		printf("Client disconnected");
		fflush(stdout);
	}
	else if(send_size == -1)
	{
		printf("recv failed");
	}
	printf("Bytes Sent: %d", send_size);
	delete [] client_message;	
	return;
}
void messageCallback(const boost::shared_ptr<topic_tools::ShapeShifter const>& msg)
{
	size_t length = ros::serialization::serializationLength(*msg);
	std::vector<uint8_t> buffer(length);

	ros::serialization::OStream ostream(&buffer[0], length);
	ros::serialization::Serializer<topic_tools::ShapeShifter>::write(ostream, *msg);

	printf("length=%d, buffer[0]=%02X,buffer[length-1]=%02X\n", buffer.size(), buffer[0],buffer[length-1]);

	sendPointData(buffer);
}

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "ros_datafeed");

	ros::NodeHandle nh;
	ros::Subscriber sub = nh.subscribe<topic_tools::ShapeShifter>("/os1_node/points", 1, &messageCallback);

	createTcpServer();
	
	pthread_t acceptThread;
	pthread_create(&acceptThread, NULL, acceptClient, NULL);

	while(true)
	{
		ros::spinOnce();
		ros::Duration(0.1).sleep();
	}

	return 0;
}
