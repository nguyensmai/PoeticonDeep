classdef Cursor
    properties
        x_pos = 0
        y_pos = 0
        wall
        x_record =[];
        y_record =[];
    end
    
    methods
        function b = Cursor(wall)
            set(gcf, 'WindowButtonMotionFcn', @Cursor.movement)
            b.wall = wall;
        end
        
        function b = update(b,record)
            pt = get(gcf, 'UserData');
            
            if ~isempty(pt)
                b.x_pos = pt(1, 1);
                b.y_pos = pt(1, 2);
            end
            
            if b.x_pos < b.wall(1)
                b.x_pos = b.wall(1);
            elseif b.x_pos > b.wall(2)
                b.x_pos = b.wall(2);
            end
            
            if b.y_pos < b.wall(3)
                b.y_pos = b.wall(3);
            elseif b.y_pos > b.wall(4)
                b.y_pos = b.wall(4);
            end
            
            if record
                b.x_record = [b.x_record b.x_pos];
                b.y_record = [b.y_record b.y_pos];
            end
        end
        
        function plot(curs)
            plot(curs.x_pos, curs.y_pos, 'bo', 'markerfacecolor', 'b', ...
                'markersize', 25)
            axis(curs.wall);
        end
    end
    
    methods (Static)
        function movement(src, evnt) %#ok<INUSD>
            pt = get(gca, 'CurrentPoint');
            set(gcf, 'UserData', pt);
        end
        
        function test_cursor()
            %%
            disp('a sample example')
            P = path();
            path(P,'../../dmp_bbo_matlab_deprecated-master_deprecated/dynamicmovementprimitive')
            h = figure;
            walls = [0 500 0 500];
            axis(walls);
            curs = Cursor(walls);
            record = 0;
            digit=-1;
            
            data=cell(9,0);
            nSamples=zeros(9,1);
            
            key_prev='?';
            key=';';
            while true
                % Update the cursor's position
                curs = update(curs,record);
                plot(curs)
                key = get(h,'CurrentCharacter');
                figure(h)
                switch lower(key)
                    case 's'
                        if key_prev~=key
                            disp('Start recording.')
                        end
                        record =1;
                        figure(h)
                    case 'e'
                        if key_prev~=key
                            disp('End recording.')
                            record = 0;
                            nSamples(digit) = nSamples(digit)+1;
                            data{digit,nSamples(digit)} = {curs.x_record,curs.y_record}; %y
                            plot( curs.x_record, curs.y_record,'-')
                            axis(walls);
                            figure(h)
                            curs.x_record=[]; curs.y_record=[];
                            pause(0.5)
                        end
                    case 'q'
                        disp('End of cycle.')
                        break;
                    otherwise
                        if isstrprop(key,'digit')
                            digit= str2num(key);
                            %disp(['Digit is ',key]);
                            
                            if key_prev~=key
                                disp(['Digit is ',key]);
                                figure(h);
                            end
                        end
                end
                key_prev=key;
                set(h,'CurrentCharacter',';')
                pause(0.01)
                save temp_cursor
            end
            
        end
        
    end
end


