classdef MovementEncoder < handle
    % Copyright (c) 2012 Sao Mai Nguyen
    %               INRIA Bordeaux - Sud-Ouest
    %               http://flowers.inria.fr
    %               e-mail: nguyensmai@gmail.com
    properties(Constant = true)
        nbParam = 20;
    end
    
    methods (Access = private, Static = true)
        function Vi = gDW(Xc,Yc,Vc,Xi,Yi,w,p,r1,r2)
            %% gaussian distance weighting
            %% OUTPUTS
            % Vi:   (mandatory) [PxQ]       gDW interpolated values
            %                                --> P=1, Q=1 yields interpolation at one
            %                                    point
            %                                --> P>1, Q=1 yields interpolation at a
            %                                    vector of points
            %                                --> P>1, Q>1 yields interpolation at a
            %                                    (ir)regular grid of points
            %% INPUTS
            % Xc:   (mandatory) [Nx1]       x coordinates of known points
            % Yc:   (mandatory) [Nx1]       y coordinates of known points
            % Vc:   (mandatory) [Nx1]       known values at [Xc, Yc] locations
            % Xi:   (mandatory) [PxQ]       x coordinates of points to be interpolated
            % Yi:   (mandatory) [PxQ]       y coordinates of points to be interpolated
            % w:    (mandatory) [scalar]    gaussian width
            % r1:   (optional)  [string]    neighbourhood type
            %                                --> 'n'     (default) number of neighbours
            %                                --> 'r'     fixed radius length
            % r2:   (optional)  [scalar]    neighbourhood size
            %                                --> number of neighbours,  if r1=='n'
            %                                    default is length(Xc)
            %                                --> radius length,         if r1=='r'
            %                                    default is largest distance between known points
            %% SYNTAX
            % ================== IDW ==================
            % all inputs:
            %   Vi = gDW(Xc,Yc,Vc,Xi,Yi,-2,'n',30);
            % 6 inputs:
            %   Vi = gDW(Xc,Yc,Vc,Xi,Yi,-2);
            %       --> r1='n'; r2=length(Xc);
            % 7 inputs:
            %   Vi = gDW(Xc,Yc,Vc,Xi,Yi,-2,'n');
            %       --> r2=length(Xc);
            %   Vi = gDW(Xc,Yc,Vc,Xi,Yi,-2,'r');
            %       --> r2=largest distance between know points [Xi,Yi] (see D1 calculation)
            % ================== SMA ==================
            %   Vi = gDW(Xc,Yc,Vc,Xi,Yi,0,'n',10);
            % ============== Spatial Map ==============
            %   Vi = gDW(Xc,Yc,Vc,Xi,Yi,-2,'n',10);
            %       -with Xi and Yi 2D arrays of coordinates relative to an (ir)regular
            %        grid.
            %% EXAMPLES
            % Interpolation at one point location:
            %   Vi = gDW([1:1:10]',[2:2:20]',rand(10,1)*100,5.5,11,-2,'n');
            % =======================================
            % Interpolation at a regular grid of unknown points:
            %   XYc = [1:1:10]';
            %   Vc = rand(10,1)*100;
            %   Xi = rand(50,50)*10;
            %   Yi = rand(50,50)*10;
            %   [Xi,Yi] = meshgrid(XYc);
            %   Vi = gDW(XYc,XYc,Vc,Xi,Yi,-2,'r',3);
            %   hold on
            %   mapshow(Xi,Yi,Vi,'DisplayType','surface')
            %   colormap gray
            %   scatter(XYc,XYc,Vc,'filled','MarkerFaceColor','g','MarkerEdgeColor','y')
            %   axis([0,11,0,11])
            %   hold off
            %% CODE
            % check consistency of input parameters
            if ~(length(Xc)-length(Yc)==0) || ~(length(Xc)-length(Vc)==0)
            whos Xc
            whos Yc
            whos Vc
            error('varargin:chk',['Vectors Xc, Yc and Vc are incorrectly sized!'])
            elseif ~(length(Xi)-length(Yi)==0)
                error('varargin:chk',['Vectors Xi and Yi are incorrectly sized!'])
            elseif nargin < 6
                error('varargin:chk',['Uncorrect number of inputs!'])
            end
            
            % build input parameters
            if nargin ~=9
                if nargin < 8   % default is 'n'
                    r1 = 'n';
                    r2 = length(Xc);
                elseif nargin==8 & r1=='n'
                    r2 = length(Xc);
                elseif nargin==8 & r1=='r'  %for 'r' default is largest distance between know points
                    [X1,X2] = meshgrid(Xc);
                    [Y1,Y2] = meshgrid(Yc);
                    D1 = sqrt((X1 - X2).^2 + (Y1 - Y2).^2);
                    r2 = max(D1(:));     % largest distance between known points
                    clear X1 X2 Y1 Y2 D1
                end
            else
                switch r1
                    case {'r', 'n'}
                        %nothing
                    otherwise
                        error('r1:chk',['Parameter r1 ("' r1 '") not properly defined!'])
                end
            end
            
            % initialize output
            Vi = zeros(size(Xi,1),size(Xi,2));
            D=[]; Vcc=[];
            
            % fixed radius
            if  strcmp(r1,'r')
                if  (r2<=0)
                    error('r2:chk','Radius must be positive!')
                    return
                end
              %  wb = waitbar(0,mfilename);
                for i=1:length(Xi(:))
             %       waitbar(i/length(Xi(:)))
                    %         if length(Xi(:))> 100, progress_bar(i, length(Xi(:)), mfilename); end
                    D    = sqrt((Xi(i)-Xc).^2 +(Yi(i)-Yc).^2);
                    indr = find(D<r2);
                    D    = D(indr);
                    Vcc  = Vc(indr);
                   % weight = exp(-w*D.^p);
                   weight = p./(p^2 + (w*D).^2);
                    if sum(weight) == 0
                        Vi(i) = Vc(1);
                    else
                        Vi(i) = sum( Vcc.*weight )/sum( weight );
                    end
                end
      
                % fixed neighbours number
            elseif  strcmp(r1,'n')
                if (r2 > length(Vc)) || (r2<1)
                    error('r2:chk','Number of neighbours not congruent with data')
                    return
                end
           %     wb = waitbar(0,mfilename);
                for i=1:length(Xi(:))
            %        waitbar(i/length(Xi(:)))
                    %         if length(Xi(:))>100, progress_bar(i, length(Xi(:)), mfilename); end
                    D = sqrt((Xi(i)-Xc).^2 +(Yi(i)-Yc).^2);
                    [D,I] = sort(D);
                    Vcc = Vc(I);
                    weight = p./(p^2 + (w*D).^2);
                   % weight = exp(-w*D(1:r2).^p);
                    if sum(weight) == 0
                        Vi(i) = Vcc(1);
                    else
                        Vi(i) = sum( Vcc(1:r2).*weight(1:r2) )/sum( weight(1:r2) );
                    end
                end
%                close(wb)
                
            end
            
            % plot3(Xc,Yc,Vc);
            % hold on
            % plot3(Xi,Yi,Vi,'o');
            % hold off
%             plot(Xi,Vi,'o')
%             hold on
%             plot(Xc,Vc,'*g')
%             hold off
            return
        end %end gDW
    end %end private static methods
    
    methods(Static = true)
        
        function [aRes,resnorm] = getMovementParameters(Xc,Vc)
            % find movement parameters from trajectoray data
            % Xc : trajectory data (time input)
            % Vc : trajectory data (value input)
            % a  : movement parameters (output)
            % resnorm: error (output)
            if size(Xc,2) ~= size(Vc,2)
                error(['MOVEMENTENCODER.getMovementParameters : error in the dimensions of input parameters', num2str(size(Xc,2)),' ~= ',num2str(size(Vc,2))]);
            end
            %Vi = MovementEncoder.gDW(Xc,Yc,Vc,Xi,Yi,-2);
            t0 = linspace(0, Xc(end), MovementEncoder.nbParam);
            y0 =linspace(Vc(1), Vc(end), MovementEncoder.nbParam); 
            a0 = [t0(2:end-1) y0(2:end-1)];
            XcVc = [Xc; Vc(1); Vc(end)];
            [a,resnorm] = lsqcurvefit(@MovementEncoder.getTrajectoryDemo,a0,XcVc,Vc)
            aRes    = [max(0,a(1:MovementEncoder.nbParam-2)) Xc(end) Vc(1) a(MovementEncoder.nbParam-1:end) Vc(end)];
        end
        
        
         function Vi = getTrajectoryDemo(paramMid,XiVi)
            % generate movement trajectory 
            % param : movement parameters (input)
            % Xc    : points (time) where the parameters are given
            % Vi    : trajectory values (output)at points (time) Xi;
            Xi = XiVi(1:end-2);
            tParam    = [0 paramMid(1:MovementEncoder.nbParam-2) Xi(end)];
            if any(tParam<0)
                Vi = 1000*ones(size(Xi));
                Vi = Vi(:);
                return
            end
            
            yParam    = [ XiVi(end-1) paramMid(MovementEncoder.nbParam-1:end) XiVi(end)];
            Xc = tParam;
            Yi = repmat(1, size(Xi));
%            Yi = repmat(1, size(Xi));
 %           Xc = linspace(0,Xi(end), MovementEncoder.nbParam);
            Yc = repmat(1, size(Xc));
            Vc = yParam;
            if size(Yc,2) ~=size(Xc,2) % MovementEncoder.nbParam
                error(['MOVEMENTENCODER.getMovementTrajectory : error in dimension of param ',num2str(size(Yc,2)),' different from ',num2str(size(Xc,2))])
            end
            sigma =  (5/(min(diff(Xc)))); %2*((MovementEncoder.nbParam)/Xi(end))^2;
            r2    =  max(diff(Xc));%Xi(end)/MovementEncoder.nbParam
            Vi    = MovementEncoder.gDW(Xc,Yc,Vc,Xi,Yi,sigma,0.5,'n',3);
       %  Vi = MovementEncoder.gDW(Xc,Yc,Vc,Xi,Yi,sigma,2,'n',3);
            Vi = Vi(:);
            
        end
        
        
        function Vi = getMovementTrajectory(param,Xi)
            % generate movement trajectory 
            % param : movement parameters (input)
            % Xc    : points (time) where the parameters are given
            % Vi    : trajectory values (output)at points (time) Xi;
         
            tParam    = [0 param(1:MovementEncoder.nbParam-1)];
            yParam    = param(MovementEncoder.nbParam:end);
            Xc = tParam;
            Yi = repmat(1, size(Xi));
%            Yi = repmat(1, size(Xi));
 %           Xc = linspace(0,Xi(end), MovementEncoder.nbParam);
            Yc = repmat(1, size(Xc));
            Vc = yParam;
            if size(Yc,2) ~=size(Xc,2) % MovementEncoder.nbParam
                error(['MOVEMENTENCODER.getMovementTrajectory : error in dimension of param ',num2str(size(Yc,2)),' different from ',num2str(size(Xc,2))])
            end
            sigma =  (5/(min(diff(Xc)))); %2*((MovementEncoder.nbParam)/Xi(end))^2;
            r2    =  max(diff(Xc));%Xi(end)/MovementEncoder.nbParam
            Vi    = MovementEncoder.gDW(Xc,Yc,Vc,Xi,Yi,sigma,0.5,'n',3);
       %  Vi = MovementEncoder.gDW(Xc,Yc,Vc,Xi,Yi,sigma,2,'n',3);
            Vi = Vi(:);
        end
        
        
        function testME
            Xc = [0:0.05:1]'
            Vc = Xc.^3
            a = MovementEncoder.getMovementParameters(Xc,Vc);
            figure
            Vinv = MovementEncoder.getMovementTrajectory(a,Xc);
            plot(Xc,Vc,'-');
            hold on
            plot([0 a(1:MovementEncoder.nbParam-1)],a(MovementEncoder.nbParam:end),'or');
            hold on
            plot(Xc,Vinv, 'X')
            
            Vc = sin(Xc*10).*Xc
            a = MovementEncoder.getMovementParameters(Xc,Vc);
            Vinv = MovementEncoder.getMovementTrajectory(a,Xc);
            figure
            plot(Xc,Vc,'-');
            hold on
            plot([0 a(1:MovementEncoder.nbParam-1)],a(MovementEncoder.nbParam:end),'or');
            hold on
            plot(Xc,Vinv, 'X')
            
        end
        
        function continuityParam
            Xc = [0:0.05:1]'
            param = 2*(rand(1,2*MovementEncoder.nbParam-1) - 0.5)
            %%
            Vinv = MovementEncoder.getMovementTrajectory(param,Xc)
            plot([0 param(1:MovementEncoder.nbParam-1)],param(MovementEncoder.nbParam:end),'or');

           % plot(linspace(0,Xc(end),4),param,'ob');
            hold on;
            plot(Xc,Vinv, 'Xb-')
            %%
            for i=1:20
                param2 = param + 0.2*(rand(1,2*MovementEncoder.nbParam-1) - 0.5);
                  Xc = [0:0.05: param2(MovementEncoder.nbParam-1)]'
                Vinv2 = MovementEncoder.getMovementTrajectory(param2,Xc);
                plot([0 param2(1:MovementEncoder.nbParam-1)],param2(MovementEncoder.nbParam:end),'or');                hold on;
                plot(Xc,Vinv2, 'Xg-')
            end
        end
        
        
        
        
    end %end methods static
    
end
% % % --------------------------------
% % % Author: Dr. Murtaza Khan
% % % Email : drkhanmurtaza@gmail.com
% % % --------------------------------
