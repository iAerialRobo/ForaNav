#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "app.h"
#include "cpx.h"
#include "commander.h"
#include "FreeRTOS.h"
#include "task.h"
#include "debug.h"
#include "log.h"
#include "param.h"

#define DEBUG_MODULE "TAKEOFF"

typedef struct {
    double x;
    double y;
    double pixel_x;
    double pixel_y;
} Coordinate;

typedef struct {
    double x;
    double y;
} Velocity;

typedef struct {
  double x;
  double y;
  double z;
} Position;

typedef struct {
    float x;
    float y;
    float vx;
    float vy;
    float P[2][2]; 
} KalmanFilter;

typedef struct {
  bool isInStraightPath; 
  double distanceToLine;  
  double nearestPointX;   
  double nearestPointY;   
} StraightPathResult;

static const float height_takeoff = 1.25f;
static const float height_land_1 = 0.5f;
static const float height_land_2 = 0.15f;
static const float distance_x = 4.2f;
static const float distance_y = -0.1f;
static const double velMax = 0.10f;
static const double velMax_mainroad = 0.1f;
static Velocity velocity = {0, 0};


static const double SameTreeTh = 0.45; 
static uint16_t numTrees = 0; 
static uint16_t numtreeBlacklist = 0;
static float x =0 ;
static float y =0 ;

static Coordinate nearestTree = {0, 0, 0, 0};
static KalmanFilter filter;
static uint16_t KalmantreeNum = 999;
static Coordinate Targetpoint = {0, 0, 0, 0};
static float dt = 0.001;
static uint16_t KalmanUpdateCount = 0;
static double real_pixel_x;
static double real_pixel_y;

bool printedReturnToPath = false;
bool printedMoveToEnd = false;
bool printedMoveToTreeTop = false;
bool checkReturnToPath = true;
bool isreplacepixel = false;
bool isUWB = false;
bool Pixeldetected = false;
void kalmanFilterInit(KalmanFilter* filter, float x, float y, float vx, float vy) {
    filter->x = x;
    filter->y = y;
    filter->vx = vx;
    filter->vy = vy;

    filter->P[0][0] = 0.1f; 
    filter->P[1][1] = 0.1f;
    filter->P[2][2] = 0.1f;
    filter->P[3][3] = 0.1f;
    filter->P[0][1] = 0.0f;
    filter->P[1][0] = 0.0f;
    KalmanUpdateCount = 0;
}

void kalmanFilterUpdate(KalmanFilter* filter, float x, float y, float dt) {

    float R = 0.1f; 

    float Kx = filter->P[0][0] / (filter->P[0][0] + R);
    float Ky = filter->P[1][1] / (filter->P[1][1] + R);

    filter->x += Kx * (x - filter->x); 
    filter->y += Ky * (y - filter->y);
    if (KalmanUpdateCount > 0) {
    	filter->vx += Kx * (x - filter->x) / (dt*KalmanUpdateCount);
    	filter->vy += Ky * (y - filter->y) / (dt*KalmanUpdateCount);
    }
    filter->P[0][0] *= (1 - Kx);
    filter->P[1][1] *= (1 - Ky);
    filter->P[2][2] *= (1 - Kx);
    filter->P[3][3] *= (1 - Ky);
    KalmanUpdateCount = 0;
}

void kalmanFilterPredict(KalmanFilter* filter, float dt) {

    filter->x += filter->vx * dt; 
    filter->y += filter->vy * dt;


    float Q = 0.02f; 
    filter->P[0][0] += Q; 
    filter->P[1][1] += Q;
    filter->P[2][2] += Q;
    filter->P[3][3] += Q;
}

static void cpxPacketCallback(const CPXPacket_t* cpxRx);

static Coordinate trees[10];
static Coordinate treeBlacklist[10];



static void setHoverSetpoint(setpoint_t *setpoint, float vx, float vy, float yaw, float z)
{
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;
  
  setpoint->mode.x = modeVelocity;
  setpoint->mode.y = modeVelocity;
  setpoint->velocity.x = vx;
  setpoint->velocity.y = vy;
  
  setpoint->mode.yaw = modeAbs;
  setpoint->attitude.yaw = yaw;
  
  setpoint->velocity_body = true;
}

static void setTakeoffSetpoint(setpoint_t *setpoint, float x, float y, float z)
{
  setpoint->mode.x = modeAbs;
  setpoint->position.x = x;
  setpoint->mode.y = modeAbs;
  setpoint->position.y = y;  
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;
  
  setpoint->velocity_body = true;
}


double drone_speed(double distance, double v_min, double v_max, double midpoint, double steepness) {
    double normalized_distance = 1.0 / (1.0 + exp(-steepness * (fabs(distance) - midpoint)));
    double speed = v_min + (v_max - v_min) * normalized_distance;
    
    if (distance < 0) {
        speed = -speed;
    }

    return speed;
}

double drone_speed2(double distance,double v_min, double v_max, double steepness) {
    double speed = steepness*fabs(distance) + 0.02;

    if (speed > v_max) {
    	speed = v_max;
    	}
    if (speed < v_min){
    	speed = v_min;
    }

    if (distance < 0) {
        speed = -speed;
    }

    return speed;
}


bool is_at_target_mainroad(Coordinate target) {
    double tolerance = 0.03;
    logVarId_t idX = logGetVarId("stateEstimate", "x");
    logVarId_t idY = logGetVarId("stateEstimate", "y");
    
    double postiion_x = logGetFloat(idX);
    double postiion_y = logGetFloat(idY);
    double distance2_x =  target.x - postiion_x;
    double distance2_y =  target.y - postiion_y;    
          
    if (fabs(postiion_x - target.x) <= tolerance &&
        fabs(postiion_y - target.y) <= tolerance) {
        return true;  
    } else{ 
        velocity.x = drone_speed(distance2_x, 0.05, 0.15, 0.1, 2);	
        velocity.y = drone_speed(distance2_y, 0.05, 0.15, 0.1, 2);
        return false;
        } 
}

bool is_at_target(Coordinate target, double error_x, double error_y) {
    double tolerance2 = 0.7;
    double tolerance3 = 40;
    logVarId_t idX = logGetVarId("stateEstimate", "x");
    logVarId_t idY = logGetVarId("stateEstimate", "y");
    
    double position_x = logGetFloat(idX);
    double position_y = logGetFloat(idY);
    
    double target_x = target.x -0.2;
    double target_y = target.y;
    
    double distance1_x =  target_x-position_x;
    double distance1_y =  target_y-position_y;
     
    if (fabs(distance1_x) >= tolerance2 ||
        fabs(distance1_y) >= tolerance2){
      	isUWB =true;
      	Pixeldetected = false;
     } 
    if (isUWB){
	velocity.x = drone_speed(distance1_x, 0.05, 0.12, 0.1, 2);    	
	velocity.y = drone_speed(distance1_y, 0.05, 0.12, 0.1, 2);    	
        DEBUG_PRINT("Control by uwb\n");
        isreplacepixel = true;	
        if (Pixeldetected){
      	isUWB =false;
    	}
        return false;

     }else{     

     if ((fabs(error_x) > tolerance3) || (fabs(error_y) > tolerance3)) {
            velocity.x = drone_speed2(error_x, 0, 0.1, 0.001);    	    		
            velocity.y = drone_speed2(error_y, 0, 0.1, 0.001);
            return false;    		    	
     } else {
    	return true;
    	}	
     
     }

        	       
}

bool isDuplicateCoordinate(double newX, double newY, double pixel_x, double pixel_y, Coordinate* trees, uint16_t numTrees, float threshold) {
    if (numTrees == 0) {
        return false; 
    }

    for (uint16_t i = 0; i < numTrees; i++) {
        float distance = sqrtf(powf(newX - trees[i].x, 2) + powf(newY - trees[i].y, 2));

        if (distance < threshold)  {
	    trees[i].pixel_x = pixel_x;
	    trees[i].pixel_y = pixel_y;
	    if (i == KalmantreeNum) {
	    real_pixel_x = pixel_x;
	    real_pixel_y = pixel_y;
	    kalmanFilterUpdate(&filter, pixel_x, pixel_y, dt);
	    Pixeldetected = true; 
	    DEBUG_PRINT("Kalman Tree %u is found, pixels update:pixel_x = %.2f, pixel_y = %.2f\n", i, trees[i].pixel_x, trees[i].pixel_y);   
	    }
            return true; 
            }
        }
    

    return false;
}

bool isDuplicateBlackCoordinate(double newX, double newY, Coordinate* trees, uint16_t numTrees, float threshold) {
    if (numTrees == 0) {
        return false;
    }

    for (uint16_t i = 0; i < numTrees; i++) {
        float distance = sqrtf(powf(newX - trees[i].x, 2) + powf(newY - trees[i].y, 2));

        if (distance < threshold)  {
            return true;
            }
        }
    

    return false; 
}


Coordinate findNearestCoordinate(const Coordinate* trees, double currentX, double currentY) {
    double minDistance = sqrt(pow(currentX - trees[0].x, 2) + pow(currentY - trees[0].y, 2));
    Coordinate nearestCoordinate = {trees[0].x, trees[0].y};
    Coordinate nearestTree = trees[0];
    for (uint16_t i = 0; i < numTrees; i++) {
        double distance = sqrt(pow(currentX - trees[i].x, 2) + pow(currentY - trees[i].y, 2));
        if (distance <= minDistance) {
            minDistance = distance;
            nearestCoordinate.x = trees[i].x;
            nearestCoordinate.y = trees[i].y;
            nearestTree = trees[i];
            KalmantreeNum = i;
        }
    }
    return nearestTree;
}

void removeCoordinate(Coordinate* trees, Coordinate coordToRemove) {
    for (uint16_t i = 0; i < numTrees; i++) {
	double distance = sqrtf(powf(coordToRemove.x - trees[i].x, 2) + powf(coordToRemove.y - trees[i].y, 2));    
        if (distance < SameTreeTh) {
            for (uint16_t j = i; j < numTrees - 1; j++) {
                trees[j] = trees[j + 1];
            }
            (numTrees)--;
            break;
        }
    }
}

void addCoordinateToBlacklist(Coordinate tree) {
    treeBlacklist[numtreeBlacklist++] = tree;  
}

static void Move_to_target(){
	static setpoint_t setpoint;
	setHoverSetpoint(&setpoint, velocity.x, velocity.y, 0, height_takeoff);
        commanderSetSetpoint(&setpoint, 3);
}

void Move_to_TreeTop() {
    logVarId_t idX = logGetVarId("stateEstimate", "x");
    logVarId_t idY = logGetVarId("stateEstimate", "y");
    double position_x = logGetFloat(idX);
    double position_y = logGetFloat(idY);
    nearestTree = findNearestCoordinate(trees, position_x, position_y);
    Coordinate nearestCoord = {nearestTree.x, nearestTree.y};
    kalmanFilterInit(&filter, nearestTree.pixel_x, nearestTree.pixel_y, 0, 0);
    real_pixel_x = nearestTree.pixel_x;
    real_pixel_y = nearestTree.pixel_y;
    if (numTrees > 0) {
        DEBUG_PRINT("fly to tree top:(%.2f,%.2f)!\n", nearestCoord.x, nearestCoord.y);
        while (1) {
            if (isreplacepixel){
            kalmanFilterInit(&filter, real_pixel_x, real_pixel_y, 0, 0);
            isreplacepixel = false;
            }	
            kalmanFilterPredict(&filter, dt);
            Targetpoint.pixel_x = filter.x;
            Targetpoint.pixel_y = filter.y;
            KalmanUpdateCount++;
            double error_x = 144-Targetpoint.pixel_x;
            double error_y = 176-Targetpoint.pixel_y;

	    if (is_at_target(nearestCoord, error_x, error_y)){
            checkReturnToPath = true;     
            DEBUG_PRINT("Arrived on tree top:(%.2f,%.2f)", nearestCoord.x, nearestCoord.y);
            removeCoordinate(trees, nearestCoord);
            addCoordinateToBlacklist(nearestCoord);
            if (numtreeBlacklist > 0){
            for (uint16_t i = 0; i < numtreeBlacklist; i++) {
DEBUG_PRINT("Tree %u: x = %.2f, y = %.2f\n", i, (double)treeBlacklist[i].x, (double)treeBlacklist[i].y);
        } 
            }			
            printedMoveToEnd = false;
            break;
            }
                                    
	    if (!printedMoveToTreeTop) {
	    DEBUG_PRINT("Move to tree top:(%.2f,%.2f)", nearestCoord.x, nearestCoord.y);
	    printedMoveToTreeTop = true;
	    }        	
            Move_to_target();
	    vTaskDelay(M2T(10));
	}        
        

    }
}

StraightPathResult checkStraightPath(Position currentPos, Position startLine, Position endLine, double deviationThreshold) {
  StraightPathResult result;
  
  double lineDirX = endLine.x - startLine.x;
  double lineDirY = endLine.y - startLine.y;

  double toStartX = currentPos.x - startLine.x;
  double toStartY = currentPos.y - startLine.y;

  double projectionLength = (toStartX * lineDirX + toStartY * lineDirY) /
                           (lineDirX * lineDirX + lineDirY * lineDirY);

  result.nearestPointX = startLine.x + projectionLength * lineDirX;
  result.nearestPointY = startLine.y + projectionLength * lineDirY;

  double distanceToLine = sqrtf(powf(currentPos.x - result.nearestPointX, 2) + powf(currentPos.y - result.nearestPointY, 2));
  result.distanceToLine = distanceToLine;

  result.isInStraightPath = (result.distanceToLine <= deviationThreshold);

  return result;
}

void appMainTask(void *param)
{
    appMain();
    vTaskSuspend(NULL);
  }

void appMain()
{
  static setpoint_t setpoint;
  static Position startPosition = {0, 0, height_takeoff};
  static Position desiredPosition = {distance_x, distance_y, height_takeoff};
  double deviationThreshold = 0.1f;
  DEBUG_PRINT("Waiting for activation in 5s...\n");
  vTaskDelay(M2T(2000));
  
  cpxRegisterAppMessageHandler(cpxPacketCallback);
  
  Coordinate target_Coordinate = {0, 0}; 
  
  paramVarId_t idPositioningDeck = paramGetVarId("deck", "bcFlow2");


  DEBUG_PRINT("Waiting for activation ...\n");

  uint8_t positioningInit = paramGetUint(idPositioningDeck);

  if (positioningInit) {
      if (1) {
      	
      	DEBUG_PRINT("Taking off after 3s ...\n");
	vTaskDelay(M2T(3000));
        vTaskDelay(M2T(10));
	logVarId_t idX = logGetVarId("stateEstimate", "x");
    	logVarId_t idY = logGetVarId("stateEstimate", "y");
	logVarId_t idZ = logGetVarId("stateEstimate", "z"); 

      	Position waypoints[] = { 
        {0.3, 0.45, height_takeoff},
        {3.9, 0.45, height_takeoff},
        {3.9, -0.45, height_takeoff},
        {0.3, -0.45, height_takeoff},
    	};
    	 
	int numWaypoints = sizeof(waypoints) / sizeof(waypoints[0]);   
	int k = 0;  
	target_Coordinate.x = waypoints[k].x;
	target_Coordinate.y = waypoints[k].y; 
        int j = 0;
        while (j < 300){
        	vTaskDelay(M2T(10));
        	setHoverSetpoint(&setpoint, 0, 0, 0, 0.8);
        	commanderSetSetpoint(&setpoint, 3);
        	j++;    
        }
        
        while (j < 600){
        	vTaskDelay(M2T(10));
        	setTakeoffSetpoint(&setpoint, 0.3, 0, height_takeoff);
        	commanderSetSetpoint(&setpoint, 3);
        	j++;    
        }      	
      	DEBUG_PRINT("Mission start! ...\n");              
        while (1) {
    		double postiion_x = logGetFloat(idX);
    		double postiion_y = logGetFloat(idY);	
		double postiion_z = logGetFloat(idZ);
		Position currentPosition = {postiion_x, postiion_y, postiion_z};

		if (!checkReturnToPath){
		startPosition = waypoints[k];
		desiredPosition = waypoints[k+1];
		StraightPathResult result = checkStraightPath(currentPosition, startPosition, desiredPosition, deviationThreshold);
		bool isInStraightPath = result.isInStraightPath;
		double nearestPointX = result.nearestPointX;
		double nearestPointY = result.nearestPointY;
		Coordinate nearestPoint = {nearestPointX, nearestPointY};
		  if (!isInStraightPath){
			if (!printedReturnToPath) {
				DEBUG_PRINT("Return to the path ...\n");
				printedReturnToPath = true;
			}
			while (1) {
			if (is_at_target_mainroad(nearestPoint)) {
			printedMoveToEnd = false;
			checkReturnToPath = true;
                	break;
                	}	
                	Move_to_target();
			vTaskDelay(M2T(10));
			}
		  }
		}
		
		printedReturnToPath = false;
		
		if (is_at_target_mainroad(target_Coordinate)) {
		DEBUG_PRINT("End point arrived, drone position is  (%.2f, %.2f, %.2f)\n", (double)currentPosition.x, (double)currentPosition.y, (double)k);
		printedMoveToEnd = false;
		k++;
		if (k > (numWaypoints-1)){
		DEBUG_PRINT("Mission completed!\n");
                	break;
                }
		target_Coordinate.x = waypoints[k].x;
		target_Coordinate.y = waypoints[k].y;
                }
                	
                Move_to_target();
                if (!printedMoveToEnd) {
        	DEBUG_PRINT("Move to the target point(%.2f, %.2f)\n",(double)target_Coordinate.x, (double)target_Coordinate.y);
        	printedMoveToEnd = true;
        	}
        	
        	Move_to_TreeTop();
        	
		vTaskDelay(M2T(10));
	}
	
        k = 0;
        while (k < 300){
        	vTaskDelay(M2T(10));
        	setHoverSetpoint(&setpoint, 0, 0, 0, height_land_1);
        	commanderSetSetpoint(&setpoint, 3);
        	k++;    
        }
        
        while (k < 600){
        	vTaskDelay(M2T(10));
        	setHoverSetpoint(&setpoint, 0, 0, 0, height_land_2);
        	commanderSetSetpoint(&setpoint, 3);
        	k++;    
        }
                
        while (1) {
        	vTaskDelay(M2T(10));     
        	memset(&setpoint, 0, sizeof(setpoint_t));
        	commanderSetSetpoint(&setpoint, 3);
        }        
      
  } else {
    DEBUG_PRINT("No flow deck installed ...\n");
  }
}
}

static void cpxPacketCallback(const CPXPacket_t* cpxRx) {
    logVarId_t idX = logGetVarId("stateEstimate", "x");
    logVarId_t idY = logGetVarId("stateEstimate", "y");
    logVarId_t idZ = logGetVarId("stateEstimate", "z");
    
    double postiion_x = logGetFloat(idX);
    double postiion_y = logGetFloat(idY);	
    double postiion_z = logGetFloat(idZ);
    
    if (numTrees < sizeof(trees) / sizeof(trees[0])) {
    	x = (144-cpxRx->data[1])*(0.0031*postiion_z)+postiion_x+0.2; 
    	y = (176-cpxRx->data[0])*(0.0031*postiion_z)+postiion_y;
     if (fabs(x) < 5.5 && fabs(y) < 2.5) {
       if (!isDuplicateBlackCoordinate(x, y, treeBlacklist, numtreeBlacklist, SameTreeTh)){
        if (!isDuplicateCoordinate(x, y, cpxRx->data[1], cpxRx->data[0], trees, numTrees, SameTreeTh)) {

            DEBUG_PRINT("Oil palm detected, drone position is  (%.2f, %.2f)\n", (double)postiion_x, (double)postiion_y);

            trees[numTrees].x = x;
            trees[numTrees].y = y;
	    trees[numTrees].pixel_x = cpxRx->data[1];
	    trees[numTrees].pixel_y = cpxRx->data[0];
	    DEBUG_PRINT("New added Oil palm location is (%.2f, %.2f), pixel_x = %.2f, pixel_y = %.2f\n", (double)x, (double)y, (double)trees[numTrees].pixel_x, (double)trees[numTrees].pixel_y);
            DEBUG_PRINT("numTrees is (%u)\n", numTrees+1);
            numTrees++; 
            
            for (uint16_t i = 0; i < numTrees; i++) {
DEBUG_PRINT("Tree %u: x = %.2f, y = %.2f, pixel_x = %.2f, pixel_y = %.2f\n", i, (double)trees[i].x, (double)trees[i].y, trees[i].pixel_x, trees[i].pixel_y);
        }            
       }
      } 
     } else {
     DEBUG_PRINT("New Oil palm detected, but coordinates out of bounds: (%.2f, %.2f)\n", (double)x, (double)y);
     }
    } else {
        DEBUG_PRINT("Tree coordinates array is full. Cannot store more trees.\n");
    }
    vTaskDelay(M2T(100));
}
