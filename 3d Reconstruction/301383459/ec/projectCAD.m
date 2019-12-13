
load('../data/PnP.mat');

PClean = estimate_pose(x, X);

[KClean, RClean, tClean] = estimate_params(PClean);

Faces=cad.faces;
Vertices=cad.vertices;

figure
trimesh(Faces,RClean(:,1),RClean(:,2),RClean(:,3))
patch('faces', Faces, 'vertices' ,Vertices);

figure 
patch('faces', Faces, 'vertices' ,Vertices);

imshow (image)

 p = patch('faces', Faces, 'vertices' ,Vertices);
    p.FaceColor='r';
    
    light                               
    daspect([1 1 1])                    
    view(3)                             
    xlabel('X'),ylabel('Y'),zlabel('Z')
   
    drawnow                