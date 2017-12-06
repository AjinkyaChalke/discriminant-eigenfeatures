function LDA()

	femaleImagesFolder = 'FaceClassification_Data/Female/';
	maleImagesFolder = 'FaceClassification_Data/Male/';
	extensionOfImages = 'TIF';

	% To select top k eigen vectors
	k = 10;

	%get images into vector
	femaleImagesVector = loadImagesFromFolderInVector( femaleImagesFolder, extensionOfImages );
	maleImagesVector = loadImagesFromFolderInVector( maleImagesFolder, extensionOfImages );

	%combine the male and female space
	imagesVector = horzcat(femaleImagesVector, maleImagesVector);

	%calculate the means
	imagesVectorMean = mean(imagesVector,2);

	%get the eigenVectors and eigenValues by performing PCA
	[eigenVectors, eigenValues] = performPCA( double(imagesVector), imagesVectorMean);

	eigenVectors = eigenVectors(:,[1 : k]);
	eigenValues = eigenValues(1 : k, :);

	modelPCA = eigenVectors'; %'

	%get the projection of male and female images
	Pn = performProjection( modelPCA, double(maleImagesVector) );
	Pp = performProjection( modelPCA, double(femaleImagesVector) );

	%get Sw
	Sw = calculateSw( Pp, Pn );

	%get Sb
	Sb = calculateSb( Pp, Pn );

	% calculating the single PC
	[V,D] = eig(Sw,Sb);	
	[~, index]=sort(diag(D),'descend');
	PC = V( :, index(1));

	% discriminant slope w
	w = PC'* modelPCA; %'

	% discriminant intercept b
	mean_Pn = mean( Pn, 2);
	mean_Pp = mean( Pp, 2);
	
	b = PC' * (mean_Pp + mean_Pn)/2 ; %'

	% Displaying various sizes
	fprintf('Displaying sizes of :-');

	fprintf('Male images vector: (%d, %d)\n', size(maleImagesVector,1),size(maleImagesVector,2));
	fprintf('Female images vector: (%d, %d)\n',size(femaleImagesVector,1),size(femaleImagesVector,2) );
	
	fprintf('PCA model: (%d, %d)\n', size(modelPCA,1),size(modelPCA,2) );

	fprintf('Projections of male images on model (Pn): (%d, %d)\n', size(Pn,1),size(Pn,2));
	fprintf('Projections of female images on model (Pp): (%d, %d)\n', size(Pp,1),size(Pp,2));
	
	fprintf('Size of Sw: (%d, %d)\n', size(Sw,1),size(Sw,2));
	fprintf('Size of Sw: (%d, %d)\n', size(Sb,1),size(Sb,2));

	fprintf('Size of LDA eigen Vectors: (%d, %d)\n', size(V,1),size(V,2));
	fprintf('Size of LDA eigen values: (%d, %d)\n', size(D,1),size(D,2));

	fprintf('Size of slope: (%d, %d)\n', size(w,1),size(w,2));
	fprintf('Size of intercept: (%d, %d)\n', size(b,1),size(b,2));

	% Plots
	figure;

	subplot(1,2,1);
	plot( eigenValues );
	title('PCA EigenValues');

	subplot(1,2,2);
	plot( D );
	title('LDA EigenValues');

	figure;
	colormap('gray');

	subplot(1,3,1);
	imagesc(reshape( imagesVectorMean,32,32));
	title('Mean Image');	

	subplot(1,3,2);
	imagesc(reshape( mean(maleImagesVector,2),32,32));
	title('Mean Male');

	subplot(1,3,3);
	imagesc(reshape( mean(femaleImagesVector,2),32,32));
	title('Mean Female');


	% Visualizing top 2 PCs before LDA
	eigenFaces = modelPCA'; %'
	figure;
	colormap('gray');
	subplot(2,2,1);
	imagesc((reshape(eigenFaces(:,1),32,32)));
	title('1st PC before LDA')

	subplot(2,2,2);
	imagesc((reshape(eigenFaces(:,10),32,32)));
	title('10th PC before LDA')

	% Visualizing PC after LDA
	pc_lda = w'; %'
	subplot(2,2,3);
	imagesc((reshape(pc_lda,32,32)));
	title('Single PC after LDA')

	figure;
	colormap('gray');
	subplot(1,2,1);
	hist( Pp );
	title( 'Histogram of Pp')

	subplot(1,2,2);
	hist( Pn );
	title( 'Histogram of Pn')

	%===============================================================================================================
	%										Using the above trained model for classification
	%===============================================================================================================
	% Lets do the classification now
	Test_femaleImagesFolder = 'FaceClassification_Data_Training/Female/';
	Test_maleImagesFolder = 'FaceClassification_Data_Training/Male/';
	fprintf('-*----*----*----*----*----*----*----*---*----*----*----*----*----*-\n');
	fprintf('Predictions\n');
	fprintf('File Name ---------------------------------------------------------> Class  Value\n');
	
	iterator_result = 1;
	error = 0;

	% Female Images
	files = dir( strcat( Test_femaleImagesFolder, strcat('*.',extensionOfImages) ) );
	for index = 1: size(files,1)
		imageFile = files(index).name;
		imageMatrix = imread(strcat(Test_femaleImagesFolder,imageFile));

		[output, class] = classifier( w, imageMatrix(:), imagesVectorMean, b );
		fprintf('%s -----> %d %d\n', strcat(Test_femaleImagesFolder,imageFile), class, output);

		result(iterator_result, 1) = 1;
		result(iterator_result, 2) = class;
		result(iterator_result, 3) = output; 
		iterator_result = iterator_result + 1;

		if 1 ~= class 
			error = error + 1;
			figure;
			colormap('gray');
			imagesc(imageMatrix);
			title('Misclassified Female Image');
		end
	end


	% Male Images
	files = dir( strcat( Test_maleImagesFolder, strcat('*.',extensionOfImages) ) );
	for index = 1: size(files,1)
		imageFile = files(index).name;
		imageMatrix = imread(strcat(Test_maleImagesFolder,imageFile));

		[output, class] = classifier( w, imageMatrix(:), imagesVectorMean, b );
		fprintf('%s -----> %d %d\n', strcat(Test_maleImagesFolder,imageFile), class, output);

		result(iterator_result, 1) = -1;
		result(iterator_result, 2) = class;
		result(iterator_result, 3) = output; 
		iterator_result = iterator_result + 1;

		if -1 ~= class 
			error = error + 1;
			figure;
			colormap('gray');
			imagesc(imageMatrix);
			title('Misclassified Male Image');
		end
	end

	fprintf(" Ground Truth   Predicted Class   Classifier Score\n");
	for index = 1 : size(result, 1)
		fprintf('     %d            %d             %d         \n', result( index, 1), result( index, 2), result( index, 3) );
	end

	total = size( result, 1);
	accuracy = (total - error) / double(total);
	fprintf('Number of correct classifications: %d\n', total - error );
	fprintf('Total number of classifications: %d\n', total);
	fprintf('Accuracy of the model: %d\n', accuracy * 100 );

	%===============================================================================================================
end

% Helper functions
% classifier
function [output, class] = classifier( w, x, u, b)
	
	output = w*(double(x) - double(u)) - b;
	
	if output < 0
		class = -1;
	else
		class = 1;
	end
end 

% function to calculate Sb
function Sb = calculateSb( Pp, Pn )
	mean_Pn = mean( Pn, 2);
	mean_Pp = mean( Pp, 2);

	size_Pn = size(Pn, 2);
	size_Pp = size(Pp, 2);


	mean_data = (size_Pn.*mean_Pn + size_Pp.*Pp)./(size_Pn + size_Pp);

	Sb = size_Pn.*((mean_Pn - mean_data)*(mean_Pn - mean_data)') + size_Pp.*((mean_Pp - mean_data)*(mean_Pp - mean_data)');
end 

% function to calculate Sw
function Sw = calculateSw( Pp, Pn )
	mean_Pn = mean( Pn, 2);
	mean_Pp = mean( Pp, 2);

	repmat_mean_Pn = repmat(mean_Pn, 1, size( Pn, 2) );
	repmat_mean_Pp = repmat(mean_Pp, 1, size( Pp, 2) );

	Sw = (Pp - repmat_mean_Pp)*(Pp - repmat_mean_Pp)' + (Pn - repmat_mean_Pn)*(Pn - repmat_mean_Pn)';
end

% function to get the images from a folder into a matrix of image vectors
function vector = loadImagesFromFolderInVector( folder, extensionOfImages )
	files = dir( strcat( folder, strcat('*.',extensionOfImages) ) );

	for index = 1: size(files,1)
		imageFile = files(index).name;
		imageMatrix = imread(strcat(folder,imageFile));
		vector(:,index) = imageMatrix(:);
	end
end

% project the images on to the model
function projections = performProjection( modelPCA, originalImages )
	meanOfOriginalImages = double(mean( originalImages, 2));
	for index = 1:size(originalImages,2)
		projections(:,index) = modelPCA * (originalImages(:,index) - meanOfOriginalImages);
	end
end

%perform PCA
function [eigenVectors, eigenValues] = performPCA( imagesVector, imagesVectorMean )
	%Calculate covariance
	covariance = ( imagesVector - imagesVectorMean )*( imagesVector-imagesVectorMean )';
	%'
	[V , D] = eig( covariance );
	eValues = diag( D );
	[eigenValues, indices] = sort( eValues, 'descend' );
	eigenVectors = V(:,indices);
end