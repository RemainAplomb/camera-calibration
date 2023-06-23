// ConsoleApplication1.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include "calibrate.h"



bool bshow = true;
static Camera cam;
void readPara(Camera &);

int main()
{
	readPara(cam);

	
    return 0;
}

