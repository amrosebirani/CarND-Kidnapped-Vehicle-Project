/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 50;
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

  for (int i=0; i<num_particles; i++) {
    Particle p = {};
    p.id = i;
    p.weight = 1.0;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
  for (int i=0; i<num_particles; i++) {
    if (fabs(yaw_rate) > 0.00001) {
      double theta_new = particles[i].theta + yaw_rate * delta_t;
      particles[i].x = particles[i].x + (velocity/yaw_rate) * (sin(theta_new)- sin(particles[i].theta));
      particles[i].y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(theta_new));
      particles[i].theta = theta_new;
    } else {
      particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
    }
    // adding noise now
    particles[i].x = particles[i].x + dist_x(gen);
    particles[i].y = particles[i].y + dist_y(gen);
    particles[i].theta = particles[i].theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (int i=0; i<observations.size(); i++) {
    double min_distance = numeric_limits<double>::max();
    int selected_id = -1;
    for (int j=0; j<predicted.size(); j++) {
      double distance = pow(observations[i].x-predicted[j].x, 2) + pow(observations[i].y-predicted[j].y, 2);
      if (distance < min_distance) {
        min_distance = distance;
        selected_id = predicted[j].id;
      }
      observations[i].id = selected_id;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  std::vector<double> nWeights;
  for (int p=0; p<particles.size(); p++) {
    Particle cp = particles[p];
    std::vector<LandmarkObs> transfomedObs;
    for (int o=0; o<observations.size(); o++) {
      LandmarkObs obs = observations[o];
      LandmarkObs t = {};
      t.x = cp.x + cos(cp.theta)*obs.x - sin(cp.theta)*obs.y;
      t.y = cp.y + sin(cp.theta)*obs.x + cos(cp.theta)*obs.y;
      t.id = obs.id;
      transfomedObs.push_back(t);
    }
    std::vector<LandmarkObs> predictedObs;
    for (int k=0; k<map_landmarks.landmark_list.size(); k++) {
      double dist = sqrt(pow(map_landmarks.landmark_list[k].x_f - cp.x, 2) + pow(map_landmarks.landmark_list[k].y_f - cp.y, 2));
      if (dist <= sensor_range) {
        LandmarkObs predictedOb = {};
        predictedOb.x = map_landmarks.landmark_list[k].x_f;
        predictedOb.y = map_landmarks.landmark_list[k].y_f;
        predictedOb.id = map_landmarks.landmark_list[k].id_i;
        predictedObs.push_back(predictedOb);
      }
    }
    dataAssociation(predictedObs, transfomedObs);
    double gauss_norm = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
    double weight = 1.0;
    for (int t=0; t<transfomedObs.size(); t++) {
      LandmarkObs ct = transfomedObs[t];
      double expox = pow(ct.x - map_landmarks.landmark_list[ct.id-1].x_f, 2)/(2*pow(std_landmark[0], 2));
      double expoy = pow(ct.y - map_landmarks.landmark_list[ct.id-1].y_f, 2)/(2*pow(std_landmark[1], 2));
      weight = weight * gauss_norm * exp(-(expox + expoy));
    }
    particles[p].weight = weight;
    nWeights.push_back(particles[p].weight);
  }
  weights = nWeights;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<Particle> rParticles;
  std::default_random_engine generator;
  std::discrete_distribution<double> distribution(weights.begin(), weights.end());
  for (int p=0; p<particles.size(); p++) {
    rParticles.push_back(particles[distribution(generator)]);
  }
  particles =rParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
