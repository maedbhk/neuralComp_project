function varargout=nc_ticaMovie(what,varargin)

% Directories
baseDir          = '/Users/maedbhking/Documents/imageica';
imageDir         ='/images';
movieDir         ='/movies';
resultDir        ='/results';



switch what
    
    case 'run'
        model=varargin{1}; % 'ica', 'isa', or 'tica'
        
        % set parameters
        [samples,winsize,rdim,dataNum]=nc_ticaMovie('setParams');
        
        % get data
        [X,whiteningMatrix,dewhiteningMatrix]=nc_ticaMovie('getImages',samples,winsize,rdim,dataNum);
        
        % get parameters for model
        p=nc_ticaMovie('getModel',model);
        
        % estimate model
        nc_ticaMovie('estimate',model,X,whiteningMatrix,dewhiteningMatrix,p)   
    case 'setParams'
        samples=10000; 
        winsize=8;
        rdim=40;
        dataNum=13;

        varargout={samples,winsize,rdim,dataNum};
    case 'getImages'
        samples=varargin{1};   % 10000 (total number of patches to take)
        winsize=varargin{2};   % 8     (patch width in pixels)
        rdim=varargin{3};      % 40    (reduced dimensionality)
        dataNum=varargin{4};   % 13    (number of images)
        
        % Output:
        % X                  the whitened data as column vectors
        % whiteningMatrix    transformation of patch-space to X-space
        % dewhiteningMatrix  inverse transformation
        
        rand('seed', 0);
        
        % This is how many patches to take per image
        getsample = floor(samples/dataNum);
        
        % Initialize the matrix to hold the patches
        X = zeros(winsize^2,getsample*dataNum);
        
        sampleNum = 1;
        for i=(1:dataNum)
            % Even things out (take enough from last image)
            if i==dataNum, getsample = samples-sampleNum+1; end
            
            % Load the image
            I = imread(fullfile(baseDir,imageDir,sprintf('%d.tiff',i)));
            
            % Normalize to zero mean and unit variance
            I = double(I);
            I = I-mean(mean(I));
            I = I/sqrt(mean(mean(I.^2)));
            
            % Sample
            fprintf('Sampling image %d...\n',i);
            [sizey sizex] = size(I);
            posx = floor(rand(1,getsample)*(sizex-winsize-2))+1;
            posy = floor(rand(1,getsample)*(sizey-winsize-1))+1;
            
            for j=1:getsample
                X(:,sampleNum) = reshape( I(posy(1,j):posy(1,j)+winsize-1, ...
                    posx(1,j):posx(1,j)+winsize-1),[winsize^2 1]);
                sampleNum=sampleNum+1;
            end
        end
        
        % Subtract local mean gray-scale value from each patch
        fprintf('Subtracting local mean...\n');
        X = X-ones(size(X,1),1)*mean(X);
        
        % Reduce the dimension and whiten at the same time!
        % Calculate the eigenvalues and eigenvectors of covariance matrix.
        fprintf ('Calculating covariance...\n');
        covarianceMatrix = X*X'/size(X,2);
        [E, D] = eig(covarianceMatrix);
        
        % Sort the eigenvalues and select subset, and whiten
        fprintf('Reducing dimensionality and whitening...\n');
        [dummy,order] = sort(diag(-D));
        E = E(:,order(1:rdim));
        d = diag(D);
        d = real(d.^(-0.5));
        D = diag(d(order(1:rdim)));
        X = D*E'*X;
        
        whiteningMatrix = D*E';
        dewhiteningMatrix = E*D^(-1);
        
        varargout={X,whiteningMatrix,dewhiteningMatrix};
        return;
    case 'getModel'
        model=varargin{1}; % options are 'tica', 'sica','ica'
        
        % PARAMETERS COMMON TO ALL ALGORITHMS
        % p.seed             random number generator seed
        % p.write            iteration interval for writing results to disk
        
        % ICA (with FastICA algorithm, tanh nonlinearity)
        % p.model            'ica'
        % p.algorithm        'fixed-point'
        % p.components       number of ICA components to estimate
        
        % ISA (gradient descent with adaptive stepsize)
        % p.model            'isa'
        % p.algorithm        'gradient'
        % p.groupsize        dimensionality of subspaces
        % p.groups           number of independent subspaces to estimate
        % p.stepsize         starting stepsize
        % p.epsi             small positive constant
        
        % TOPOGRAPHIC ICA (gradient descent with adaptive stepsize)
        % p.model            'tica'
        % p.algorithm        'gradient'
        % p.xdim             columns in map
        % p.ydim             rows in map
        % p.maptype          'standard' or 'torus'
        % p.neighborhood     'ones3by3' (only one implemented so far)
        % p.stepsize         starting stepsize
        % p.epsi             small positive constant
        
        switch model,
            
            case 'tica'
                % model for complex cells + topography
                p.seed = 1;
                p.write = 5;
                p.model = 'tica';
                p.algorithm = 'gradient';
                p.xdim = 8;
                p.ydim = 5;
                p.maptype = 'torus';
                p.neighborhood = 'ones3by3';
                p.stepsize = 0.1;
                p.epsi = 0.005;
            case 'isa'
                % complex cell
                p.seed = 1;
                p.write = 5;
                p.model = 'isa';
                p.algorithm = 'gradient';
                p.groupsize = 2;
                p.groups = 20;
                p.stepsize = 0.1;
                p.epsi = 0.005;
            case 'ica'
                p.seed = 1;
                p.write = 5;
                p.model = 'ica';
                p.algorithm = 'fixed-point';
                p.components = 40;
                
        end
        
        varargout={p};
    case 'estimate'
        model=varargin{1}; % 'ica', 'isa', or 'tica'
        X=varargin{2};
        whiteningMatrix=varargin{3}; 
        dewhiteningMatrix=varargin{4}; 
        p=varargin{5};

        N = size(X,2);
        
        % Initialize the random number generator.
        rand('seed',p.seed);
        
        switch model, % take random initialise vectors
            case 'ica'
                B = randn(size(X,1),p.components);
            case 'isa'
                B = randn(size(X,1),p.groupsize*p.groups);
            case 'tica'
                B = randn(size(X,1),p.xdim*p.ydim);
        end

        % ...and decorrelate (=orthogonalize in whitened space)
        B = B*real((B'*B)^(-0.5));
        n = size(B,2);
        
        % START THE ITERATION...
        % Print the time when started (and save along with parameters).
        c=clock;
        if c(5)<10, timestarted = ['Started at: ' int2str(c(4)) ':0' int2str(c(5))];
        else timestarted = ['Started at: ' int2str(c(4)) ':' int2str(c(5))];
        end
        fprintf([timestarted '\n']);
        p.timestarted = timestarted;
        
        % Initialize iteration counter
        iter=0;
        
        % Loop forever, writing result periodically
        while 1
            
            % Increment iteration counter
            iter = iter+1;
            fprintf('(%d)',iter);
            
            switch model,
                case 'ica'
                    if strcmp(p.model,'ica') & strcmp(p.algorithm,'fixed-point')
                        
                        % This is tanh but faster than matlabs own version
                        hypTan = 1 - 2./(exp(2*(X'*B))+1);
                        
                        % This is the fixed-point step
                        B = X*hypTan/N - ones(size(B,1),1)*mean(1-hypTan.^2).*B;
                        
                    end
                case 'isa'
                    if strcmp(p.algorithm,'gradient')
                        
                        % Calculate linear filter responses and their squares
                        U = B'*X; Usq = U.^2;
                        
                        % For each subspace
                        for i=1:p.groups
                            
                            % These are the columns of B making up the subspace
                            cols = (i-1)*p.groupsize+(1:p.groupsize);
                            
                            % Calculate nonlinearity of subspace energy
                            g = -((p.epsi + sum(U(cols,:).^2)).^(-0.5));
                            
                            % Calculate gradient
                            dB(:,cols) = X*(U(cols,:).*(ones(p.groupsize,1)*g))'/N;
                            
                        end
                    end
                case 'tica'
                    if strcmp(p.algorithm,'gradient')
                        
                        % Neighborhood matrix: NB(i,j) = strength of unit j in neighb. of
                        % unit i. In addition, we will create a matrix NBNZ, which gives
                        % the positions of the non-zero entries in NB, to lower the
                        % computational expenses.
                        fprintf('Generating neighborhood matrix...\n');
                        [NBNZ,NB] = GenerateNB(p);
                        
                        % Calculate linear filter responses and their squares
                        U = B'*X; Usq = U.^2;
                        
                        % Calculate local energies
                        for ind=1:n
                            E(ind,:) = NB(ind,NBNZ{ind}) * Usq(NBNZ{ind},:);
                        end
                        
                        % Take nonlinearity
                        g = -((p.epsi + E).^(-0.5));
                        
                        % Calculate convolution with neighborhood
                        for ind=1:n
                            F(ind,:) = NB(ind,NBNZ{ind}) * g(NBNZ{ind},:);
                        end
                        
                        % This is the total gradient
                        dB = X*(U.*F)'/N;
                        
                    end
            end
            
            % ADAPT STEPSIZE FOR GRADIENT ALGORITHMS
            if strcmp(p.algorithm,'gradient'),
                % Use starting stepsize for gradient methods
                stepsize = p.stepsize;
                obj = [];
                objiter = [];
                
                % Perform this adaptation only every 5 steps
                if rem(iter,5)==0 | iter==1
                    
                    % Take different length steps
                    Bc{1} = B + 0.5*stepsize*dB;
                    Bc{2} = B + 1.0*stepsize*dB;
                    Bc{3} = B + 2.0*stepsize*dB;
                    
                    % Orthogonalize each one
                    for i=1:3, Bc{i} = Bc{i}*real((Bc{i}'*Bc{i})^(-0.5)); end
                    
                    % Calculate objective values in each case
                    for i=1:3
                        switch model,
                            case 'isa'
                                Usq = (Bc{i}'*X).^2;
                                for ind=1:p.groups,
                                    cols = (ind-1)*p.groupsize+(1:p.groupsize);
                                    E(ind,:) = sum(Usq(cols,:));
                                end
                                objective(i) = mean(mean(sqrt(p.epsi+E)));
                            case 'tica'
                                Usq = (Bc{i}'*X).^2;
                                for ind=1:n, E(ind,:)= NB(ind,NBNZ{ind}) * Usq(NBNZ{ind},:); end
                                objective(i) = mean(mean(sqrt(p.epsi+E)));
                        end
                    end
                    
                    % Compare objective values, pick step giving minimum
                    if objective(1)<objective(2) & objective(1)<objective(3)
                        % Take shorter step
                        stepsize = stepsize/2;
                        fprintf('Stepsize now: %.4f\n',stepsize);
                        obj = [obj objective(1)];
                    elseif objective(1)>objective(3) & objective(2)>objective(3)
                        % Take longer step
                        stepsize = stepsize*2;
                        fprintf('Stepsize now: %.4f\n',stepsize);
                        obj = [obj objective(3)];
                    else
                        % Don't change step
                        obj = [obj objective(2)];
                    end
                    
                    objiter = [objiter iter];
                    fprintf('\nObjective value: %.6f\n',obj(end));
                    
                end
            end
            
            B = B + stepsize*dB;

            % Ortogonalize (equal to decorrelation since we are
            % in whitened space)
            B = B*real((B'*B)^(-0.5));

            % Write the results to disk
            fname=fullfile(baseDir,resultDir,sprintf('%s.mat',model)); 
            if rem(iter,p.write)==0 | iter==1
                A = dewhiteningMatrix * B;
                W = B' * whiteningMatrix;
                
                fprintf(['Writing file: ' fname '...']);
                if strcmp(p.algorithm,'gradient')
                    eval(['save ' fname ' W A p iter obj objiter']);
                else
                    eval(['save ' fname ' W A p iter']);
                end
                fprintf(' Done!\n'); 
            end
        end     
    case 'visualise'
        model=varargin{1}; 
        mag=varargin{2};
        cols=varargin{3}; 
        
        % visual - display a basis for image patches
        % A        the basis, with patches as column vectors
        % mag      magnification factor
        % cols     number of columns (x-dimension of map)
        
        load(fullfile(baseDir,resultDir,sprintf('%s.mat',model)))
        
        % Get maximum absolute value (it represents white or black; zero is gray)
        maxi=max(max(abs(A)));
        mini=-maxi;
        
        % This is the side of the window
        dim = sqrt(size(A,1));
        
        % Helpful quantities
        dimm = dim-1;
        dimp = dim+1;
        rows = size(A,2)/cols;
        if rows-floor(rows)~=0, error('Fractional number of rows!'); end
        
        % Initialization of the image
        I = maxi*ones(dim*rows+rows-1,dim*cols+cols-1);
        
        for i=0:rows-1
            for j=0:cols-1
                
                % This sets the patch
                I(i*dimp+1:i*dimp+dim,j*dimp+1:j*dimp+dim) = ...
                    reshape(A(:,i*cols+j+1),[dim dim]);
            end
        end
        
        I = imresize(I,mag);
        
        figure;
        colormap(gray(256));
        iptsetpref('ImshowBorder','tight');
%         subplot('position',[100,100,10,10]);
        imshow(I,[mini maxi]);
        truesize;
    case 'getMovie'
        
       V=VideoReader(fullfile(baseDir,movieDir,'V6_Birdman.mov'));
       
       V.CurrentTime=0.5; 
       
       currAxes=axes; 
       
       while hasFrame(V)
           vidFrame=readFrame(V);
           image(vidFrame,'Parent',currAxes);
           currAxes.Visible='off';
           pause(1/V.FrameRate); 
       end
       
       disp(x); 
       
end



%-----------------------------------------------------------------
% LOCAL FUNCTIONS
%-----------------------------------------------------------------

function dircheck(dir) % make a new directory
if ~exist(dir,'dir');
    warning('%s doesn''t exist. Creating one now. You''re welcome! \n',dir);
    mkdir(dir);
end

%-----------------------------------------------------------------
% GenerateNB - generates the neighborhood matrix for TICA
%-----------------------------------------------------------------
function [NBNZ,NB] = GenerateNB(p)

% This will hold the neighborhood function entries
NB = zeros(p.xdim*p.ydim*[1 1]);

% This is currently the only implemented neighborhood
if strcmp(p.neighborhood,'ones3by3')==0
    error('No such neighborhood allowed!');
end

% Step through nodes one at a time to build the matrix
ind = 0;
for y=1:p.ydim
    for x=1:p.xdim
        
        ind = ind+1;
        
        % Rectangular neighbors
        [xn,yn] = meshgrid( (x-1):(x+1), (y-1):(y+1) );
        xn = reshape(xn,[1 9]);
        yn = reshape(yn,[1 9]);
        
        if strcmp(p.maptype,'torus')
            
            % Cycle round
            i = find(yn<1); yn(i)=yn(i)+p.ydim;
            i = find(yn>p.ydim); yn(i)=yn(i)-p.ydim;
            i = find(xn<1); xn(i)=xn(i)+p.xdim;
            i = find(xn>p.xdim); xn(i)=xn(i)-p.xdim;
            
        elseif strcmp(p.maptype,'standard')
            
            % Take only valid nodes
            i = find(yn>=1 & yn<=p.ydim & xn>=1 & xn<=p.xdim);
            xn = xn(i);
            yn = yn(i);
            
        else
            error('No such map type!');
        end
        
        % Set neighborhood
        NB( ind, (yn-1)*p.xdim + xn )=1;
        
    end
end

% For each unit, calculate the non-zero columns!
for i=1:p.xdim*p.ydim
    NBNZ{i} = find(NB(i,:));
end