%CITS4402 Project 1 - GUI
%Amos Viray (23729527), Cameron Waddingham (23737222)
classdef GUI < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                  matlab.ui.Figure
        GridLayout                matlab.ui.container.GridLayout
        NextImageButton           matlab.ui.control.StateButton
        PreviousImageButton       matlab.ui.control.StateButton
        GridLayout2               matlab.ui.container.GridLayout
        PredictionEditField       matlab.ui.control.EditField
        PredictionEditFieldLabel  matlab.ui.control.Label
        LoadDirectoryButton       matlab.ui.control.Button
        UIAxes                    matlab.ui.control.UIAxes
    end

    
    properties (Access = public)
        imagefile %Array with image file paths
        currentindex = 1 %Image number tracker
        svmodel %Trained model from project_1.m
        predictions %Array for predictions
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: LoadDirectoryButton
        function LoadDirectoryButtonPushed(app, event)
            
            dirpath = uigetdir;
            if dirpath ~= 0
                data = load('svmodel.mat'); % Load model
                app.svmodel = data.svmodel;

                files = dir(fullfile(dirpath, '*'));
                files = files(~[files.isdir]);
                
                app.imagefile = fullfile(dirpath, {files.name});
                app.predictions = zeros(size(app.imagefile));  % numeric array for binary predictions
                app.currentindex = 1;
                
                img = imread(app.imagefile{app.currentindex});
                imshow(img, 'Parent', app.UIAxes);
                %Perform HOG extraction and prediction on current image
                features = computeHOG(img);
                [label, ~] = predict(app.svmodel, features');
                app.PredictionEditField.Value = string(label);
            end
        end

        % Value changed function: PreviousImageButton
        function PreviousImageButtonValueChanged(app, event)
            value = app.PreviousImageButton.Value;
            app.currentindex = app.currentindex - 1;
            if app.currentindex < 1
                app.currentindex = numel(app.imagefile); %Loop back to final image
                
            end
            img = imread(app.imagefile{app.currentindex});
            imshow(img, 'Parent', app.UIAxes);
            app.PreviousImageButton.Value = false;
            %Perform HOG extraction and prediction on current image after
            %button is pressed
            features = computeHOG(img);
            [label, ~] = predict(app.svmodel, features');
            app.PredictionEditField.Value = string(label);
        end

        % Value changed function: NextImageButton
        function NextImageButtonValueChanged(app, event)
            value = app.NextImageButton.Value;
            app.currentindex = app.currentindex + 1;
            if app.currentindex > numel(app.imagefile)
                app.currentindex = 1; %Loop back to starting image
                
            end
            img = imread(app.imagefile{app.currentindex});
            imshow(img, 'Parent', app.UIAxes);
            app.NextImageButton.Value = false;
            %Perform HOG extraction and prediction on current image after
            %button is pressed
            features = computeHOG(img);
            [label, ~] = predict(app.svmodel, features');
            app.PredictionEditField.Value = string(label);


            %Save prediction (binary)
            app.predictions(app.currentindex) = label;  % assuming label is 0 or 1
            
            %Prepare data table for saving
            T = table(app.imagefile(:), app.predictions(:), 'VariableNames', {'Filename', 'Prediction'});
            
            %Write to Excel file (overwrite every time)
            writetable(T, 'predictions.xlsx');
        end

        % Value changed function: PredictionEditField
        function PredictionEditFieldValueChanged(app, event)
            value = app.PredictionEditField.Value;
            
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'MATLAB App';

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {'1x', '2x', '1x'};
            app.GridLayout.RowHeight = {'1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x'};

            % Create UIAxes
            app.UIAxes = uiaxes(app.GridLayout);
            app.UIAxes.XAxisLocation = 'origin';
            app.UIAxes.XColor = 'none';
            app.UIAxes.XTick = [];
            app.UIAxes.YAxisLocation = 'origin';
            app.UIAxes.YColor = 'none';
            app.UIAxes.YTick = [];
            app.UIAxes.ZColor = 'none';
            app.UIAxes.Layout.Row = [1 5];
            app.UIAxes.Layout.Column = 2;

            % Create LoadDirectoryButton
            app.LoadDirectoryButton = uibutton(app.GridLayout, 'push');
            app.LoadDirectoryButton.ButtonPushedFcn = createCallbackFcn(app, @LoadDirectoryButtonPushed, true);
            app.LoadDirectoryButton.Layout.Row = 7;
            app.LoadDirectoryButton.Layout.Column = 2;
            app.LoadDirectoryButton.Text = 'Load Directory';

            % Create GridLayout2
            app.GridLayout2 = uigridlayout(app.GridLayout);
            app.GridLayout2.RowHeight = {'1x'};
            app.GridLayout2.Layout.Row = 6;
            app.GridLayout2.Layout.Column = 2;

            % Create PredictionEditFieldLabel
            app.PredictionEditFieldLabel = uilabel(app.GridLayout2);
            app.PredictionEditFieldLabel.HorizontalAlignment = 'center';
            app.PredictionEditFieldLabel.Layout.Row = 1;
            app.PredictionEditFieldLabel.Layout.Column = 1;
            app.PredictionEditFieldLabel.Text = 'Prediction';

            % Create PredictionEditField
            app.PredictionEditField = uieditfield(app.GridLayout2, 'text');
            app.PredictionEditField.ValueChangedFcn = createCallbackFcn(app, @PredictionEditFieldValueChanged, true);
            app.PredictionEditField.Editable = 'off';
            app.PredictionEditField.Layout.Row = 1;
            app.PredictionEditField.Layout.Column = 2;

            % Create PreviousImageButton
            app.PreviousImageButton = uibutton(app.GridLayout, 'state');
            app.PreviousImageButton.ValueChangedFcn = createCallbackFcn(app, @PreviousImageButtonValueChanged, true);
            app.PreviousImageButton.Text = 'Previous Image';
            app.PreviousImageButton.Layout.Row = 3;
            app.PreviousImageButton.Layout.Column = 1;

            % Create NextImageButton
            app.NextImageButton = uibutton(app.GridLayout, 'state');
            app.NextImageButton.ValueChangedFcn = createCallbackFcn(app, @NextImageButtonValueChanged, true);
            app.NextImageButton.Text = 'Next Image';
            app.NextImageButton.Layout.Row = 3;
            app.NextImageButton.Layout.Column = 3;

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = GUI

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end