clear
%% Generate samples (with whitening and dewhiteningMatrix)

cd 'C:\BoxSync\Dropbox\Berkeley\NeuralComp tICA Project\imageica\code'
% Get  5  samples of  100x100  window images
% Select only 160 dimensions (Principle components for whitening)
[X, whiteningMatrix, dewhiteningMatrix] = data( 50000, 100, 160 );

save dp_test_tica dewhiteningMatrix whiteningMatrix X
%% Estimate A  

clear
cd 'C:\BoxSync\Dropbox\Berkeley\NeuralComp tICA Project\imageica\code'

load dp_test_tica

p.seed = 1;
p.write = 5;
p.model = 'tica';
p.algorithm = 'gradient';
p.xdim = 16;
p.ydim = 10;
p.maptype = 'torus';
p.neighborhood = 'ones3by3';
p.stepsize = 0.1;
p.epsi = 0.005;
estimate( whiteningMatrix, dewhiteningMatrix, '..\results\tica_dp.mat', p, 300, X);

%% Visualise the bases
clear
cd 'C:\BoxSync\Dropbox\Berkeley\NeuralComp tICA Project\imageica\code'
load 'C:\BoxSync\Dropbox\Berkeley\NeuralComp tICA Project\imageica\results\tica_dp.mat'


mag = 1;
visual( A, mag, 16 )

%% Get frames
% For this example we are actually going to get three windows from a static
% image

clear
cd 'C:\BoxSync\Dropbox\Berkeley\NeuralComp tICA Project\imageica\code'

% Load the image
I = imread(['../data/1.tiff']);


%% Visualise tiff image
mag = 1;
I = imresize(I,mag);
maxi=max(max(abs(I)));
mini=-maxi;

figure;
colormap(gray(256));
iptsetpref('ImshowBorder','tight');
subplot('position',[0,0,1,1]);
imshow(I,[mini maxi]);
truesize;

%% select just the top left(?) window

getsample = 3;
winsize = 100;
sampleNum = 1;

% Initialize the matrix to hold the patches
X_frames = zeros(winsize^2,getsample);

posx = [1 21 41];
posy = [1 1 1];

for j=1:getsample
    X_frames(:,j) = reshape( I(posy(1,j):posy(1,j)+winsize-1, ...
        posx(1,j):posx(1,j)+winsize-1),[winsize^2 1]);
end

%% Visualise frames (X)

mag = 1;

frameNum = 1;

frame = X_frames(:,frameNum);

frame = reshape(frame, 100, 100);

maxi=max(max(abs(frame)));
% mini=-maxi; % Maybe use this version for visualising bases
mini=0;  

figure;
colormap(gray(256));
iptsetpref('ImshowBorder','tight');
subplot('position',[0,0,1,1]);
imshow(frame,[mini maxi]);
truesize;
%% Calculate Activation 

load dp_test_tica

% frame number to plot
frameNum = 1

% Raw frame as column of pixels
xframe = X_frames(:,frameNum);

% Whiten frame
whitenedFrame = whiteningMatrix * xframe;

% Whiten A
B = whiteningMatrix * A;

% Find activation s
s = inv(B) * whitenedFrame;
  

% Plot activations
activations = reshape(s, 16, 10)';

activations = imresize(activations, 16,'nearest' );

maxi=max(max(abs(activations)));
mini=-maxi;

figure;
colormap(gray(256));
iptsetpref('ImshowBorder','tight');
subplot('position',[0,0,1,1]);
imshow(activations,[mini maxi]);
truesize;

